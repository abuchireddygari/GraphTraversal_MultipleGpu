/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/*THIS FILE HAS BEEN MODIFIED FROM THE ORIGINAL*/

/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This work was (partially) funded by the DARPA XDATA program under
AFRL Contract #FA8750-13-C-0002.

This material is based upon work supported by the Defense Advanced
Research Projects Agency (DARPA) under Contract No. D14PC00029.
*/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/util/operators.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>
#include <cuda.h>
using namespace b40c;
using namespace graph;

namespace GASengine
{
  namespace vertex_centric
  {
    namespace contract_atomic
    {

      /**
       * Templated texture reference for visited mask
       */
      template<typename VisitedMask>
      struct BitmaskTex
      {
        static texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> ref;
      };
      template<typename VisitedMask>
      texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> BitmaskTex<VisitedMask>::ref;

      /**
       * CTA tile-processing abstraction for BFS frontier contraction
       */
      template<typename KernelPolicy, typename Program>
      struct Cta
      {
        //---------------------------------------------------------------------
        // Typedefs and Constants
        //---------------------------------------------------------------------

        typedef typename KernelPolicy::VertexId VertexId;
        typedef typename KernelPolicy::ValidFlag ValidFlag;
        typedef typename KernelPolicy::VisitedMask VisitedMask;
        typedef typename KernelPolicy::SizeT SizeT;
        typedef typename Program::DataType EValue;
        typedef typename Program::MiscType MiscType;
        typedef typename Program::GatherType GatherType;
        typedef typename Program::VertexType VertexType;
        typedef typename Program::EdgeType EdgeType;

        typedef typename KernelPolicy::RakingDetails RakingDetails;
        typedef typename KernelPolicy::SmemStorage SmemStorage;

        //---------------------------------------------------------------------
        // Members
        //---------------------------------------------------------------------

        // Input and output device pointers
        VertexId *d_in; // Incoming edge frontier
        VertexId *d_out; // Outgoing vertex frontier
        MiscType *d_predecessor_in; // Incoming predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
        GatherType *m_gatherTmp;
        VertexType &vertex_list;
        EdgeType &edge_list;
        char* d_bitmap_visited;
        VisitedMask *d_visited_mask; // Mask for detecting visited status

        // Work progress
        VertexId iteration; // Current BFS iteration
        VertexId queue_index; // Current frontier queue counter index
        util::CtaWorkProgress &work_progress; // Atomic workstealing and queueing counters
        SizeT max_vertex_frontier; // Maximum size (in elements) of outgoing vertex frontier
        int num_gpus; // Number of GPUs
        int selector;
        SizeT *d_frontier_size;

        // Operational details for raking scan grid
        RakingDetails raking_details;

        // Shared memory for the CTA
        SmemStorage &smem_storage;

        // Whether or not to perform bitmask culling (incurs extra latency on small frontiers)
        bool bitmask_cull;

        //---------------------------------------------------------------------
        // Helper Structures
        //---------------------------------------------------------------------

        /**
         * Tile of incoming edge frontier to process
         */
        template<
        int LOG_LOADS_PER_TILE,
                int LOG_LOAD_VEC_SIZE>
                struct Tile
        {
          //---------------------------------------------------------------------
          // Typedefs and Constants
          //---------------------------------------------------------------------

          enum
          {
            LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE,
            LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE
          };

          //---------------------------------------------------------------------
          // Members
          //---------------------------------------------------------------------

          // Dequeued vertex ids
          VertexId vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
          VertexId predecessor_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

          // Whether or not the corresponding vertex_id is valid for exploring
          ValidFlag flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

          // Global scatter offsets
          SizeT ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];

          //---------------------------------------------------------------------
          // Helper Structures
          //---------------------------------------------------------------------

          /**
           * Iterate over vertex id
           */
          template<int LOAD, int VEC, int dummy = 0 >
                  struct Iterate
          {

            /**
             * InitFlags
             */
            static __device__ __forceinline__ void InitFlags(Tile * tile)
            {
              // Initially valid if vertex-id is valid
              tile->flags[LOAD][VEC] = (tile->vertex_id[LOAD][VEC] == -1) ? 0 : 1;
              tile->ranks[LOAD][VEC] = tile->flags[LOAD][VEC];

              // Next
              Iterate<LOAD, VEC + 1 > ::InitFlags(tile);
            }

            /**
             * BitmaskCull
             */
            static __device__ __forceinline__ void BitmaskCull(
                                                               Cta *cta,
                                                               Tile * tile)
            {
              if (tile->vertex_id[LOAD][VEC] != -1)
              {

                // Location of mask byte to read
                SizeT mask_byte_offset = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

                // Bit in mask byte corresponding to current vertex id
                VisitedMask mask_bit = 1 << (tile->vertex_id[LOAD][VEC] & 7);

                // Read byte from from visited mask in tex
                VisitedMask tex_mask_byte = tex1Dfetch(
                                                       BitmaskTex<VisitedMask>::ref,
                                                       mask_byte_offset);

                if (mask_bit & tex_mask_byte)
                {

                  // Seen it
                  tile->vertex_id[LOAD][VEC] = -1;

                }
                else
                {

                  VisitedMask mask_byte;
                  //						util::io::ModifiedLoad<util::io::ld::cg>::Ld(
                  //							mask_byte, cta->d_visited_mask + mask_byte_offset);
                  mask_byte = cta->d_visited_mask[mask_byte_offset];

                  mask_byte |= tex_mask_byte;

                  if (mask_bit & mask_byte)
                  {

                    // Seen it
                    tile->vertex_id[LOAD][VEC] = -1;

                  }
                  else
                  {

                    // Update with best effort
                    mask_byte |= mask_bit;
                    //							util::io::ModifiedStore<util::io::st::cg>::St(
                    //								mask_byte,
                    //								cta->d_visited_mask + mask_byte_offset);

                    //							printf("cta->d_visited_mask + mask_byte_offset=%d, mask_byte=%d\n", (int)cta->d_visited_mask + mask_byte_offset, (int)mask_byte);
                    //							printf("CTA contraction: %d %d\n", sizeof(B40C_REG8), sizeof(B40C_CAST8));
                    //							asm volatile ("st.global.cg.u8 [%0], %1;" : : "l"( (VisitedMask*)(cta->d_visited_mask + mask_byte_offset) ), "h"((unsigned short)mask_byte));

                    //							printf("bidx=%d, tidx=%d, cta->d_visited_mask=%d, mask_byte_offset=%d\n", blockIdx.x, threadIdx.x, (int)cta->d_visited_mask, mask_byte_offset);
                    cta->d_visited_mask[mask_byte_offset] = mask_byte;
                  }
                }
              }

              // Next
              Iterate<LOAD, VEC + 1 > ::BitmaskCull(cta, tile);
            }

            /**
             * VertexCull
             */
            static __device__ __forceinline__ void VertexCull(
                                                              Cta *cta,
                                                              Tile * tile)
            {
              if (tile->vertex_id[LOAD][VEC] != -1)
              {
                typename Program::contract contract_functor;
                contract_functor(cta->iteration, cta->d_bitmap_visited, tile->vertex_id[LOAD][VEC], cta->vertex_list, cta->edge_list, cta->m_gatherTmp, tile->predecessor_id[LOAD][VEC]);
              }

              // Next
              Iterate<LOAD, VEC + 1 > ::VertexCull(cta, tile);
            }

            /**
             * HistoryCull
             */
            static __device__ __forceinline__ void HistoryCull(
                                                               Cta *cta,
                                                               Tile * tile)
            {
              if (tile->vertex_id[LOAD][VEC] != -1)
              {

                int hash = ((typename KernelPolicy::UnsignedBits) tile->vertex_id[LOAD][VEC]) % SmemStorage::HISTORY_HASH_ELEMENTS;
                VertexId retrieved = cta->smem_storage.history[hash];

                if (retrieved == tile->vertex_id[LOAD][VEC])
                {
                  // Seen it
                  tile->vertex_id[LOAD][VEC] = -1;

                }
                else
                {
                  // Update it
                  cta->smem_storage.history[hash] = tile->vertex_id[LOAD][VEC];
                }
              }

              // Next
              Iterate<LOAD, VEC + 1 > ::HistoryCull(cta, tile);
            }

            /**
             * WarpCull
             */
            static __device__ __forceinline__ void WarpCull(
                                                            Cta *cta,
                                                            Tile * tile)
            {
              if (tile->vertex_id[LOAD][VEC] != -1)
              {

                int warp_id = threadIdx.x >> 5;
                int hash = tile->vertex_id[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

                cta->smem_storage.state.vid_hashtable[warp_id][hash] = tile->vertex_id[LOAD][VEC];
                VertexId retrieved = cta->smem_storage.state.vid_hashtable[warp_id][hash];

                if (retrieved == tile->vertex_id[LOAD][VEC])
                {

                  cta->smem_storage.state.vid_hashtable[warp_id][hash] = threadIdx.x;
                  VertexId tid = cta->smem_storage.state.vid_hashtable[warp_id][hash];
                  if (tid != threadIdx.x)
                  {
                    tile->vertex_id[LOAD][VEC] = -1;
                  }
                }
              }

              // Next
              Iterate<LOAD, VEC + 1 > ::WarpCull(cta, tile);
            }
          };

          /**
           * Iterate next load
           */
          template<int LOAD, int dummy>
                  struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
          {
            // InitFlags

            static __device__ __forceinline__ void InitFlags(Tile * tile)
            {
              Iterate < LOAD + 1, 0 > ::InitFlags(tile);
            }

            // BitmaskCull

            static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile * tile)
            {
              Iterate < LOAD + 1, 0 > ::BitmaskCull(cta, tile);
            }

            // VertexCull

            static __device__ __forceinline__ void VertexCull(Cta *cta, Tile * tile)
            {
              Iterate < LOAD + 1, 0 > ::VertexCull(cta, tile);
            }

            // HistoryCull

            static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile * tile)
            {
              Iterate < LOAD + 1, 0 > ::HistoryCull(cta, tile);
            }

            // WarpCull

            static __device__ __forceinline__ void WarpCull(Cta *cta, Tile * tile)
            {
              Iterate < LOAD + 1, 0 > ::WarpCull(cta, tile);
            }
          };

          /**
           * Terminate iteration
           */
          template<int dummy>
                  struct Iterate<LOADS_PER_TILE, 0, dummy>
          {
            // InitFlags

            static __device__ __forceinline__ void InitFlags(Tile * tile)
            {
            }

            // BitmaskCull

            static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile * tile)
            {
            }

            // VertexCull

            static __device__ __forceinline__ void VertexCull(Cta *cta, Tile * tile)
            {
            }

            // HistoryCull

            static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile * tile)
            {
            }

            // WarpCull

            static __device__ __forceinline__ void WarpCull(Cta *cta, Tile * tile)
            {
            }
          };

          //---------------------------------------------------------------------
          // Interface
          //---------------------------------------------------------------------

          /**
           * Initializer
           */
          __device__ __forceinline__ void InitFlags()
          {
            Iterate < 0, 0 > ::InitFlags(this);
          }

          /**
           * Culls vertices based upon whether or not we've set a bit for them
           * in the d_visited_mask bitmask
           */
          __device__ __forceinline__ void BitmaskCull(Cta * cta)
          {
            Iterate < 0, 0 > ::BitmaskCull(cta, this);
          }

          /**
           * Culls vertices
           */
          __device__ __forceinline__ void VertexCull(Cta * cta)
          {
            Iterate < 0, 0 > ::VertexCull(cta, this);
          }

          /**
           * Culls duplicates within the warp
           */
          __device__ __forceinline__ void WarpCull(Cta * cta)
          {
            Iterate < 0, 0 > ::WarpCull(cta, this);
          }

          /**
           * Culls duplicates within recent CTA history
           */
          __device__ __forceinline__ void HistoryCull(Cta * cta)
          {
            Iterate < 0, 0 > ::HistoryCull(cta, this);
          }
        };

        //---------------------------------------------------------------------
        // Methods
        //---------------------------------------------------------------------

        /**
         * Constructor
         */
        __device__ __forceinline__ Cta(
                                       VertexId iteration,
                                       VertexId queue_index,
                                       int num_gpus,
                                       int selector,
                                       SizeT* d_frontier_size,
                                       SmemStorage &smem_storage,
                                       VertexId *d_in,
                                       VertexId *d_out,
                                       MiscType *d_predecessor_in,
                                       GatherType *m_gatherTmp,
                                       VertexType &vertex_list,
                                       EdgeType &edge_list,
                                       char* d_bitmap_visited,
                                       VisitedMask *d_visited_mask,
                                       util::CtaWorkProgress &work_progress,
                                       SizeT max_vertex_frontier) :

        iteration(iteration),
        queue_index(queue_index),
        num_gpus(num_gpus),
        selector(selector),
        d_frontier_size(d_frontier_size),
        raking_details(smem_storage.state.raking_elements, smem_storage.state.warpscan, 0),
        smem_storage(smem_storage),
        d_in(d_in),
        d_out(d_out),
        d_predecessor_in(d_predecessor_in),
        m_gatherTmp(m_gatherTmp),
        vertex_list(vertex_list),
        edge_list(edge_list),
        d_bitmap_visited(d_bitmap_visited),
        d_visited_mask(d_visited_mask),
        work_progress(work_progress),
        max_vertex_frontier(max_vertex_frontier),
        bitmask_cull(
                     (KernelPolicy::END_BITMASK_CULL < 0) ?
                     true : // always bitmask cull
                     (KernelPolicy::END_BITMASK_CULL == 0) ?
                     false : // never bitmask cull
                     (iteration < KernelPolicy::END_BITMASK_CULL))
        {
          // Initialize history duplicate-filter
          for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS)
          {
            smem_storage.history[offset] = -1;
          }
        }

        /**
         * Process a single, full tile
         */
        __device__ __forceinline__ void ProcessTile(
                                                    SizeT cta_offset,
                                                    const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
        {
          Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

          //            printf("Doing contract processtile: threadIdx.x=%d, guarded_elements=%d\n", threadIdx.x, guarded_elements);
          // Load tile
          util::io::LoadTile<
                  KernelPolicy::LOG_LOADS_PER_TILE,
                  KernelPolicy::LOG_LOAD_VEC_SIZE,
                  KernelPolicy::THREADS,
                  KernelPolicy::QUEUE_READ_MODIFIER,
                  false > ::LoadValid(
                                      tile.vertex_id,
                                      d_in,
                                      cta_offset,
                                      guarded_elements,
                                      (VertexId) - 1);

          // Load predecessor vertices as well
          util::io::LoadTile<
                  KernelPolicy::LOG_LOADS_PER_TILE,
                  KernelPolicy::LOG_LOAD_VEC_SIZE,
                  KernelPolicy::THREADS,
                  KernelPolicy::QUEUE_READ_MODIFIER,
                  false > ::LoadValid(
                                      tile.predecessor_id,
                                      d_predecessor_in,
                                      cta_offset,
                                      guarded_elements);

          tile.VertexCull(this); // using vertex visitation status (update discovered vertices)

          // Cull visited vertices and update discovered vertices
          //		if (bitmask_cull) {
          tile.BitmaskCull(this); // using global visited mask
          //		}

          // Cull duplicates using local CTA collision hashing
          tile.HistoryCull(this);

          // Cull duplicates using local warp collision hashing
          tile.WarpCull(this);

          // Init valid flags and ranks
          tile.InitFlags();

          // Protect repurposable storage that backs both raking lanes and local cull scratch
          __syncthreads();

          // Scan tile of ranks, using an atomic add to reserve
          // space in the contracted queue, seeding ranks
          util::Sum<SizeT> scan_op;
          SizeT new_queue_offset = util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
                                                                                                                     raking_details,
                                                                                                                     tile.ranks,
                                                                                                                     &d_frontier_size[selector^1],
                                                                                                                     scan_op);

          //          if (threadIdx.x < (num_gpus - 1))
          //          {
          //            printf("Doing contract: blockIdx.x=%d, threadIdx.x=%d, num_gpus=%d\n", blockIdx.x, threadIdx.x, num_gpus);

          //          printf("Contract: blockIdx.x=%d, selector=%d, d_frontier_size[%d]=%d\n", blockIdx.x, selector, selector^1, d_frontier_size[selector^1]);

          // Check updated queue offset for overflow due to redundant expansion
          if (new_queue_offset >= max_vertex_frontier)
          {
            if (blockIdx.x == 0 && threadIdx.x == 0)
              printf("Frontier queue overflow.  Please increase queue-sizing factor.\n");
            work_progress.SetOverflow<SizeT > ();
            util::ThreadExit();
          }

          // Scatter directly (without first contracting in smem scratch), predicated
          // on flags
          util::io::ScatterTile<
                  KernelPolicy::LOG_LOADS_PER_TILE,
                  KernelPolicy::LOG_LOAD_VEC_SIZE,
                  KernelPolicy::THREADS,
                  KernelPolicy::QUEUE_WRITE_MODIFIER>::Scatter(
                                                               d_out,
                                                               tile.vertex_id,
                                                               tile.flags,
                                                               tile.ranks);
          //          }
        }
      };

    } // namespace contract_atomic
  } // namespace vertex_centric
} // namespace GASengine

