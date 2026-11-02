/*
  Copyright 2026 SINTEF Digital.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <opm/grid/cpgrid/refinement/conforming/ConformingBlockBuilder.hpp>

#include <opm/grid/CpGrid.hpp>
#include <opm/grid/cpgrid/CpGridData.hpp>
#include <opm/grid/cpgrid/refinement/conforming/EdgeConformal.hpp>
#include <opm/grid/cpgrid/refinement/GridStateWriter.hpp>
#include <opm/grid/cpgrid/refinement/conforming/LeafGridAssembler.hpp>
#include <opm/grid/cpgrid/refinement/conforming/LevelGridAssembler.hpp>

#include <opm/common/ErrorMacros.hpp>

#include <dune/common/parallel/communication.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Opm
{
namespace Refinement
{

ConformingBlockBuilder::ConformingBlockBuilder(const std::array<int,3>& parentDims,
                                               std::vector<double> coord,
                                               std::vector<double> zcorn,
                                               std::vector<int> actnum,
                                               bool edgeConformal)
{
    auto input = std::make_shared<RetainedCornerPointInput>();
    input->dims = parentDims;
    input->coord = std::move(coord);
    input->zcorn = std::move(zcorn);
    input->actnum = std::move(actnum);
    input->edgeConformal = edgeConformal;
    input_ = std::move(input);
}

ConformingBlockBuilder::ConformingBlockBuilder(std::shared_ptr<const RetainedCornerPointInput> input)
    : input_(std::move(input))
{
}

namespace
{

// Corner-point description of one grid level (index 0 = level zero, index b+1 =
// the level grid built for requests[b]): a nested box is refined from its parent
// LGR's own description. A view: the data stays where it is.
struct LevelGeom
{
    std::array<int,3> dims{};
    const double* coord{nullptr};
    const double* zcorn{nullptr};
    const int* actnum{nullptr};
};

// Per-box presence on this rank (distributed runs). The rank-interior model: a box is either fully owned here (all its cells
// local and interior) or absent (no cell local, not even as overlap).
// Anything in between means the box crosses a rank boundary / touches the
// overlap, which is not supported.
enum class BoxPresence { Owned, Absent };

BoxPresence classifyBox(const Dune::cpgrid::CpGridData& level0,
                        const std::array<int,3>& dims,
                        const Opm::Refinement::BlockRefinement& req)
{
    int interior = 0;
    int nonInterior = 0;
    const auto& globalCell = level0.globalCell();
    for (int c = 0; c < level0.size(0); ++c) {
        const int cart = globalCell[c];
        const int i = cart % dims[0];
        const int j = (cart / dims[0]) % dims[1];
        const int k = cart / (dims[0]*dims[1]);
        if (i >= req.startIJK[0] && i < req.endIJK[0]
            && j >= req.startIJK[1] && j < req.endIJK[1]
            && k >= req.startIJK[2] && k < req.endIJK[2]) {
            const Dune::cpgrid::Entity<0> e(level0, c, true);
            if (e.partitionType() == Dune::InteriorEntity) {
                ++interior;
            } else {
                ++nonInterior;
            }
        }
    }
    // Presence is judged by what this rank holds, not by counting the box: a
    // rank that owns every box cell it can see, and sees no box cell it does not
    // own, holds the whole box -- a split box puts the far side in this rank's
    // overlap, which is exactly the non-interior case. Counting instead needs a
    // total that is right for the distributed grid, and the two candidates are
    // both wrong: the Cartesian extent ignores inactive cells, and the retained
    // ACTNUM predates MINPV. On Norne those two errors are 623 and 97 cells.
    if (interior == 0 && nonInterior == 0) {
        return BoxPresence::Absent;
    }
    if (interior > 0 && nonInterior == 0) {
        return BoxPresence::Owned;
    }
    OPM_THROW(std::logic_error, "Refinement box '" + req.name + "' is split across MPI ranks or "
                           "touches the overlap region. Only rank-interior LGRs are supported; "
                           "use CpGrid::setPartitionCellGroups to keep the box on one rank.");
}

// Cells of the parent grid inside the box: the hosts left after processing
// (inactive cells and cells MINPV removes are not in the grid).
int presentHostCells(const Dune::cpgrid::CpGridData& parent,
                     const std::array<int,3>& dims,
                     const Opm::Refinement::BlockRefinement& req)
{
    int present = 0;
    const auto& globalCell = parent.globalCell();
    for (int c = 0; c < parent.size(0); ++c) {
        const int cart = globalCell[c];
        const int i = cart % dims[0];
        const int j = (cart / dims[0]) % dims[1];
        const int k = cart / (dims[0]*dims[1]);
        if (i >= req.startIJK[0] && i < req.endIJK[0]
            && j >= req.startIJK[1] && j < req.endIJK[1]
            && k >= req.startIJK[2] && k < req.endIJK[2]) {
            ++present;
        }
    }
    return present;
}

// refine-before-redistribute classification: before load balancing the global
// grid lives on rank 0 (all cells) and is empty on the other ranks. Classify a
// box purely by cell *presence* (ignoring partition type, which is not yet set):
// the rank that holds the box's cells refines it; ranks with none carry an
// empty placeholder level grid. A host missing on the rank that holds the grid
// is inactive or removed by MINPV, not on another rank.
BoxPresence boxCellPresence(const Dune::cpgrid::CpGridData& level0,
                            const std::array<int,3>& dims,
                            const Opm::Refinement::BlockRefinement& req)
{
    return (presentHostCells(level0, dims, req) > 0) ? BoxPresence::Owned
                                                     : BoxPresence::Absent;
}

} // anonymous namespace

void ConformingBlockBuilder::build(Dune::CpGrid& grid,
                                   const std::vector<BlockRefinement>& requests)
{
    if (grid.maxLevel() != 0) {
        OPM_THROW(std::logic_error, "ConformingBlockBuilder requires an unrefined starting grid.");
    }
    // Touching boxes need no conformity check here. Corners on a shared face or
    // edge coincide bitwise where the subdivisions agree, being resampled from
    // the same parent description by the same arithmetic, and are merged by the
    // leaf assembly. Boxes sharing a face with different in-face subdivisions get
    // a sub-face mosaic when one side is uniformly finer by whole multiples, and
    // the overlay of both subdivisions otherwise. Boxes that meet only at an edge
    // or a corner share no face, so different subdivisions along it just leave
    // nodes of one side on the edge (the leaf grid is face-conformal). Graded
    // boxes may still not touch another box (checked in the leaf assembly).

    auto& storage = grid.currentData();
    // The refinement is "distributed" (rank-interior: only the owning rank
    // refines each box) only once the grid has actually been scattered. Before
    // load balancing - the refine-before-redistribute path - the grid is still
    // the replicated global grid; refine it serially on every rank instead, so
    // the builder refines every cell (no rank-interior classifyBox) and the
    // leaf assembly does no collectives, producing the full refined grid a
    // serial run yields, ready to be distributed afterwards.
    const bool distributed = grid.comm().size() > 1 && grid.isDistributed();
    // Capture the grid's actual (world) communicator before any new level or
    // leaf grid is pushed - CpGrid::comm() reads the back() entry of the
    // storage, which is the coarse level-zero grid at this point and therefore
    // still carries the world communicator on every rank.
    const auto worldComm = grid.comm();
    // Use a self-communicator for serial refinement (refine-before-redistribute
    // and the self-communicator output grid) so no operation is collective
    // over the world communicator; the distributed simulation grid keeps the
    // world communicator.
    Dune::MPIHelper::MPICommunicator comm = Dune::MPIHelper::getLocalCommunicator();
    if (distributed) {
        comm = grid.comm();
    }
    // refine-before-redistribute: comm().size() > 1 but the grid is not yet
    // scattered. The global grid is on rank 0 and empty on the others, so we
    // classify boxes by cell presence (rank 0 refines, the rest get empty
    // placeholders) and refine serially with the self-communicator above.
    const bool refineBefore = grid.comm().size() > 1 && !grid.isDistributed();

    // Parent corner-point description per built level (index 0 = GLOBAL base,
    // index b+1 = the level grid of request b). A nested box is refined from
    // its parent level's resampled geometry, not the global grid; it rides on
    // its parent's rank in both parallel models, and the leaf assembly enforces
    // its containment.
    const auto& dims = input_->dims;
    const int* actnum0 = input_->actnum.empty() ? nullptr : input_->actnum.data();
    std::vector<LevelGeom> levelGeom(requests.size() + 1);
    levelGeom[0] = LevelGeom{ dims, input_->coord.data(), input_->zcorn.data(), actnum0 };
    // Each refined level's own description, for the boxes nested in it.
    std::vector<RefinedBlockGrdecl> refinedGeom(requests.size());
    std::map<std::string,int> nameToLevel{ {"GLOBAL", 0} };
    // Per-box presence (refine-before-redistribute), keyed by box name, so a
    // nested box can inherit its parent box's presence (see below).
    std::map<std::string,BoxPresence> presenceOf;

    // Per-rank construction of the level grids. In a distributed run this can
    // throw on one rank (e.g. classifyBox rejecting a box that ends up touching
    // the overlap region) while other ranks succeed and march on into the leaf
    // assembly's collectives (cc.max / cc.allgather). Those would then block
    // forever waiting for the rank that already threw -- the classic asymmetric
    // exception deadlock. Run the loop under a collective-safe guard below.
    const auto buildLevelGrids = [&]() {
        for (std::size_t b = 0; b < requests.size(); ++b) {
            const auto& req = requests[b];
            const auto itParent = nameToLevel.find(req.parentGridName);
            if (itParent == nameToLevel.end()) {
                OPM_THROW(std::logic_error, "Refinement box '" + req.name + "' names parent grid '"
                    + req.parentGridName + "', which has not been built yet. Parent LGRs "
                    "must be ordered before their children.");
            }
            const int parentLevel = itParent->second;
            const LevelGeom& pg = levelGeom[parentLevel];

            // In a distributed run, only the rank that owns the (rank-interior)
            // box refines it; other ranks carry an empty placeholder level grid.
            // Nested boxes in a distributed run are still serial-only (guarded
            // above), so a distributed run is always top-level here and classifies
            // against level zero. The refine-before-redistribute path puts the full
            // global grid on a single rank and classifies a top-level box by its
            // cell presence there. A NESTED box addresses cells in its parent LGR's
            // own (local) Cartesian space, so it cannot be classified against the
            // global grid; instead it inherits its parent box's presence - a child
            // fully contained in its parent lives on whatever rank refines the
            // parent (and the parent-before-child ordering guarantees the parent's
            // presence is already known). Plain serial stays Owned.
            const bool nested = (req.parentGridName != "GLOBAL");
            BoxPresence presence = BoxPresence::Owned;
            if (distributed) {
                // Rank-interior model: a nested box's I/J/K are parent-LGR-local, so
                // it cannot be classified against level zero. A contained child lives
                // on whichever rank owns its parent box, so inherit the parent's
                // classification (parent processed before child).
                presence = nested ? presenceOf.at(req.parentGridName)
                                  : classifyBox(*storage[0], dims, req);
            } else if (refineBefore) {
                presence = nested ? presenceOf.at(req.parentGridName)
                                  : boxCellPresence(*storage[0], dims, req);
            }
            // A box with no host left (all inactive, or removed by MINPV) has
            // nothing to refine: it becomes an empty LGR, as on a rank that
            // does not hold it. The corner-point processing below needs at
            // least one cell.
            if ((presence == BoxPresence::Owned)
                && (presentHostCells(*storage[parentLevel], pg.dims, req) == 0)) {
                presence = BoxPresence::Absent;
            }
            presenceOf[req.name] = presence;
            RefinedBlockGrdecl& childRefined = refinedGeom[b];
            auto level = (presence == BoxPresence::Owned)
                ? assembleBlockLevelGrid(*storage[parentLevel], pg.dims,
                                         pg.coord, pg.zcorn, pg.actnum,
                                         req, static_cast<int>(b) + 1, parentLevel,
                                         storage, comm, &childRefined)
                : assembleEmptyLevelGrid(req, static_cast<int>(b) + 1, storage, comm);
            storage.push_back(std::move(level));
            levelGeom[b + 1] = LevelGeom{ childRefined.dims,
                                          childRefined.coord.data(),
                                          childRefined.zcorn.data(),
                                          childRefined.actnum.empty() ? nullptr
                                                                      : childRefined.actnum.data() };
            nameToLevel[req.name] = static_cast<int>(b) + 1;
        }
    };

    if (distributed) {
        // Collective-safe: catch a per-rank failure, agree across ranks with a
        // single reduction, and throw symmetrically (with the offending rank's
        // message) so the run fails cleanly instead of deadlocking in the leaf
        // assembly's collectives below.
        int localBuildError = 0;
        std::string localBuildMsg;
        try {
            buildLevelGrids();
        } catch (const std::exception& e) {
            localBuildError = 1;
            localBuildMsg = e.what();
        }
        Dune::Communication<Dune::MPIHelper::MPICommunicator> cc(comm);
        if (cc.max(localBuildError) != 0) {
            OPM_THROW(std::runtime_error, (localBuildError != 0
                ? ("Parallel LGR build failed on rank " + std::to_string(cc.rank())
                   + ": " + localBuildMsg)
                : ("Parallel LGR build aborted on rank " + std::to_string(cc.rank())
                   + " because another rank failed (see that rank's message).")));
        }
    } else {
        buildLevelGrids();
    }

    std::shared_ptr<Dune::cpgrid::CpGridData> leaf =
        assembleLeafGrid(storage, requests, dims,
                         input_->coord.data(), input_->zcorn.data(), actnum0,
                         comm);
    if (input_->edgeConformal) {
        edgeConformalizeLeaf(*leaf);
    }
    storage.push_back(std::move(leaf));

    if (refineBefore) {
        // The new level grids (indices 1..N) and the leaf (back) were assembled
        // with a self-communicator so the serial refinement does no collectives.
        // Restore the world communicator on them: CpGrid::comm() reads the leaf,
        // so without this the refined grid would report a size-1 communicator
        // and the subsequent load balancing would treat it as a serial grid and
        // skip distributing the refined leaf. Level zero (index 0) is the
        // pre-existing coarse grid and already carries the world communicator.
        for (std::size_t i = 1; i < storage.size(); ++i) {
            GridStateWriter::setCommunicator(*storage[i], worldComm);
        }
    }
}

} // namespace Refinement
} // namespace Opm
