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

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>
#include <GASengine/enactor_vertex_centric.cuh>

using namespace b40c;
using namespace graph;

#define FRONTIER_RATIO 100000000

namespace GASengine
{
  namespace vertex_centric
  {
    namespace expand_atomic_flag
    {

      /**
       * CTA tile-processing abstraction for frontier expansion
       */
      template<typename SizeT>
      struct RowOffsetTex
      {
        static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
      };
      template<typename SizeT>
      texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;

      /**
       * Derivation of KernelPolicy that encapsulates tile-processing routines
       */
      template<typename KernelPolicy, typename Program>
      struct Cta
      {

        /**
         * Helper device functions
         */

        //CTA reduction
        template<typename T>
        static __device__                                    __forceinline__ T CTAReduce(T* partial)
        {
          for (size_t s = KernelPolicy::THREADS / 2; s > 0; s >>= 1)
          {
            if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
            __syncthreads();
          }
          return partial[0];
        }

        //Warp reduction
        template<typename T>
        static __device__                                    __forceinline__ T WarpReduce(T* partial, size_t warp_id)
        {
          for (size_t s = B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) / 2; s > 0; s >>= 1)
          {
            if (threadIdx.x < warp_id * B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + s) partial[threadIdx.x] += partial[threadIdx.x + s];
            __syncthreads();
          }
          return partial[warp_id * B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)];
        }

        //---------------------------------------------------------------------
        // Typedefs
        //---------------------------------------------------------------------

        typedef typename KernelPolicy::VertexId VertexId;
        typedef typename KernelPolicy::SizeT SizeT;
        typedef typename Program::VertexType VertexType;
        typedef typename Program::EdgeType EdgeType;

        typedef typename KernelPolicy::SmemStorage SmemStorage;

        typedef typename KernelPolicy::SoaScanOp SoaScanOp;
        typedef typename KernelPolicy::RakingSoaDetails RakingSoaDetails;
        typedef typename KernelPolicy::TileTuple TileTuple;

        typedef typename KernelPolicy::CoarseGrid CoarseGrid;
        typedef typename KernelPolicy::FineGrid FineGrid;

        typedef util::Tuple<SizeT (*)[KernelPolicy::LOAD_VEC_SIZE], SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> RankSoa;

        //---------------------------------------------------------------------
        // Members
        //---------------------------------------------------------------------

        // Input and output device pointers
        VertexId *d_in;						// Incoming vertex frontier
        VertexId *d_out;						// Outgoing edge frontier
        VertexId *d_predecessor_out;			// Outgoing predecessor edge frontier
        VertexId *d_column_indices;			// CSR column-indices array
        SizeT *d_row_offsets;				// CSR row-offsets array
        VertexType vertex_list;
        EdgeType edge_list;
        VertexId *d_edgeCSC_indices;
        int* d_active_flags;
        char* d_changed;

        // Work progress
        VertexId queue_index;				// Current frontier queue counter index
        util::CtaWorkProgress &work_progress;				// Atomic workstealing and queueing counters
        SizeT max_edge_frontier;			// Maximum size (in elements) of outgoing edge frontier
        int num_gpus;					// Number of GPUs
        int iteration;
        SizeT num_elements;
        SizeT* deviceMappedValueEdge;
        int selector;
        int previous_frontier_size;

        // Operational details for raking grid
        RakingSoaDetails raking_soa_details;

        // Shared memory for the CTA
        SmemStorage &smem_storage;

        //---------------------------------------------------------------------
        // Helper Structures
        //---------------------------------------------------------------------

        /**
         * Tile of incoming vertex frontier to process
         */
        template<int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
        struct Tile
        {
          //---------------------------------------------------------------------
          // Typedefs and Constants
          //---------------------------------------------------------------------

          enum
          {
            LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE, LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE
          };

          typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;

          //---------------------------------------------------------------------
          // Members
          //---------------------------------------------------------------------

          // Dequeued vertex ids
          VertexId vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

          // Edge list details
          SizeT row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
          SizeT row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

          // Global scatter offsets.  Coarse for CTA/warp-based scatters, fine for scan-based scatters
          SizeT fine_count;
          SizeT coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
          SizeT fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

          // Progress for expanding scan-based gather offsets
          SizeT row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
          SizeT progress;

          //---------------------------------------------------------------------
          // Helper Structures
          //---------------------------------------------------------------------

          /**
           * Iterate next vector element
           */
          template<int LOAD, int VEC, int dummy = 0>
          struct Iterate
          {
            /**
             * Init
             */
            template<typename Tile>
            static __device__ __forceinline__ void Init(Tile *tile)
            {
              tile->row_length[LOAD][VEC] = 0;
              tile->row_progress[LOAD][VEC] = 0;

              Iterate<LOAD, VEC + 1>::Init(tile);
            }

            /**
             * Inspect
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
            {
              if (tile->vertex_id[LOAD][VEC] != -1)
              {

                // Translate vertex-id into local gpu row-id (currently stride of num_gpu)
                VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

                // Load neighbor row range from d_row_offsets
                Vec2SizeT row_range;
                row_range.x = cta->d_row_offsets[row_id];
                row_range.y = cta->d_row_offsets[row_id + 1];
//                    row_range.x = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
//                    row_range.y = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1);

                // Node is previously unvisited: compute row offset and length
                tile->row_offset[LOAD][VEC] = row_range.x;
                tile->row_length[LOAD][VEC] = row_range.y - row_range.x;
              }

              tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ? tile->row_length[LOAD][VEC] : 0;

              tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ? 0 : tile->row_length[LOAD][VEC];

              Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
            }

            /**
             * Expand by CTA
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
            {

              // CTA-based expansion/loading
              while (true)
              {

                // Vie
                if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD)
                {
                  cta->smem_storage.state.cta_comm = threadIdx.x;
                }

                __syncthreads();

                // Check
                int owner = cta->smem_storage.state.cta_comm;
                if (owner == KernelPolicy::THREADS)
                {
                  // No contenders
                  break;
                }

                if (owner == threadIdx.x)
                {

                  // Got control of the CTA: command it
                  cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
                  cta->smem_storage.state.warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];									// queue rank
                  cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
                  cta->smem_storage.state.warp_comm[0][3] = tile->vertex_id[LOAD][VEC];									// predecessor

                  // Unset row length
                  tile->row_length[LOAD][VEC] = 0;

                  // Unset my command
                  cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;	// invalid
                }

                __syncthreads();

                // Read commands
                SizeT coop_offset = cta->smem_storage.state.warp_comm[0][0];
                SizeT coop_rank = cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
                SizeT coop_oob = cta->smem_storage.state.warp_comm[0][2];

                VertexId row_id = cta->smem_storage.state.warp_comm[0][3];

                VertexId neighbor_id_tmp;

//                typename Program::expand_vertex expand_vertex_functor;
//                bool changed = expand_vertex_functor(row_id, cta->d_changed[row_id], cta->vertex_list, cta->edge_list);
                bool changed = cta->d_changed[row_id];

                while (coop_offset + threadIdx.x < coop_oob)
                {
                  SizeT col_idx = coop_offset + threadIdx.x;
                  // Gather
//                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);
                  neighbor_id_tmp = cta->d_column_indices[col_idx];

                  int edge_id;
                  if (cta->d_edgeCSC_indices)
                    edge_id = cta->d_edgeCSC_indices[col_idx];
                  else
                    edge_id = col_idx;

                  VertexId frontier;
                  typename Program::MiscType misc_value;
                  typename Program::expand_edge expand_edge_functor;

                  if (changed && atomicCAS(&cta->d_active_flags[neighbor_id_tmp], 0, 1) == 0)
                  {
                    expand_edge_functor(true, cta->iteration, row_id, neighbor_id_tmp, edge_id, cta->vertex_list, cta->edge_list, frontier, misc_value);
                  }
                  else
                  {
                    frontier = -1;
                  }

                  cta->d_out[cta->smem_storage.state.coarse_enqueue_offset + coop_rank] = frontier;

                  coop_offset += KernelPolicy::THREADS;
                  coop_rank += KernelPolicy::THREADS;
                }

              }

              // Next vector element
              Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
            }

            /**
             * Expand by warp
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
            {
              if (KernelPolicy::WARP_GATHER_THRESHOLD < KernelPolicy::CTA_GATHER_THRESHOLD)
              {

                // Warp-based expansion/loading
                int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                int lane_id = util::LaneId();

                while (__any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD))
                {

                  if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)
                  {
                    // Vie for control of the warp
                    cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
                  }

                  if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0])
                  {

                    // Got control of the warp
                    cta->smem_storage.state.warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];									// start
                    cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
                    cta->smem_storage.state.warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
                    cta->smem_storage.state.warp_comm[warp_id][3] = tile->vertex_id[LOAD][VEC];								// predecessor

                    // Unset row length
                    tile->row_length[LOAD][VEC] = 0;
                  }

                  SizeT coop_offset = cta->smem_storage.state.warp_comm[warp_id][0];
                  SizeT coop_rank = cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
                  SizeT coop_oob = cta->smem_storage.state.warp_comm[warp_id][2];

                  VertexId row_id = cta->smem_storage.state.warp_comm[warp_id][3];
//                      VertexId predecessor_id = row_id;
                  VertexId neighbor_id_tmp;

//                  typename Program::expand_vertex expand_vertex_functor;
//                  bool changed = expand_vertex_functor(row_id, cta->d_changed[row_id], cta->vertex_list, cta->edge_list);
                  bool changed = cta->d_changed[row_id];
//                      while (coop_offset + B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) < coop_oob)

                  while (coop_offset + lane_id < coop_oob)
                  {
                    // Gather
//                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, cta->d_column_indices + coop_offset + lane_id);
                    SizeT col_idx = coop_offset + lane_id;
                    neighbor_id_tmp = cta->d_column_indices[col_idx];

                    VertexId frontier;
                    typename Program::MiscType misc_value;
                    typename Program::expand_edge expand_edge_functor;

                    int edge_id;
                    if (cta->d_edgeCSC_indices)
                      edge_id = cta->d_edgeCSC_indices[col_idx];
                    else
                      edge_id = col_idx;

                    if (changed && atomicCAS(&cta->d_active_flags[neighbor_id_tmp], 0, 1) == 0)
                      expand_edge_functor(true, cta->iteration, row_id, neighbor_id_tmp, edge_id, cta->vertex_list, cta->edge_list, frontier, misc_value);
                    else
                      frontier = -1;

                    cta->d_out[cta->smem_storage.state.coarse_enqueue_offset + coop_rank] = frontier;
                    cta->d_predecessor_out[cta->smem_storage.state.coarse_enqueue_offset + coop_rank] = misc_value;

                    coop_offset += B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                    coop_rank += B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                  }
                }

                // Next vector element
                Iterate<LOAD, VEC + 1>::ExpandByWarp(cta, tile);
              }
            }

            /**
             * Expand by scan
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
            {
              // Attempt to make further progress on this dequeued item's neighbor
              // list if its current offset into local scratch is in range
              SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;
//                  if(blockIdx.x == 0 && threadIdx.x == 0)
//                	  printf("sizeof(SmemStorage)=%d, sizeof(state)=%d, MAX_SCRATCH_BYTES_PER_CTA=%d, GATHER_ELEMENTS=%d, SCRATCH_ELEMENT_SIZE=%d, PARENT_ELEMENTS=%d\n",
//                  				sizeof(SmemStorage), sizeof(SmemStorage::State),
//                  				SmemStorage::MAX_SCRATCH_BYTES_PER_CTA,
//                  				SmemStorage::GATHER_ELEMENTS,
//                  				SmemStorage::SCRATCH_ELEMENT_SIZE,
//                  				SmemStorage::PARENT_ELEMENTS);

              while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) && (scratch_offset < SmemStorage::GATHER_ELEMENTS))
              {
                // Put gather offset into scratch space
                cta->smem_storage.gather_offsets[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

//                    if (cta->smem_storage.gather_offsets[scratch_offset] < -1 || cta->smem_storage.gather_offsets[scratch_offset] >= 28143065
////                    		|| (blockIdx.x == 0 && threadIdx.x == 0)
//                    				)
////                    if(cta->smem_storage.gather_offsets[scratch_offset] < -1)
//                    {
//                       printf("ExpandByScan: bidx=%d, tidx=%d, scratch_offset=%d, gather_offsets=%d, shared_ptr=%lld\n", blockIdx.x, threadIdx.x, scratch_offset, cta->smem_storage.gather_offsets[scratch_offset], cta->smem_storage.gather_offsets);
//                    }

                // Put dequeued vertex as the predecessor into scratch space
                cta->smem_storage.gather_predecessors[scratch_offset] = tile->vertex_id[LOAD][VEC];

                tile->row_progress[LOAD][VEC]++;
                scratch_offset++;
              }

              //otherwise it's a repeating visit to old vertex, do nothing

              // Next vector element
              Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
            }
          };

          /**
           * Iterate next load
           */
          template<int LOAD, int dummy>
          struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
          {
            /**
             * Init
             */
            template<typename Tile>
            static __device__ __forceinline__ void Init(Tile *tile)
            {
              Iterate<LOAD + 1, 0>::Init(tile);
            }

            /**
             * Inspect
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
            {
              Iterate<LOAD + 1, 0>::Inspect(cta, tile);
            }

            /**
             * Expand by CTA
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
            {
              Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
            }

            /**
             * Expand by warp
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
            {
              Iterate<LOAD + 1, 0>::ExpandByWarp(cta, tile);
            }

            /**
             * Expand by scan
             */
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
            {
              Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
            }
          };

          /**
           * Terminate
           */
          template<int dummy>
          struct Iterate<LOADS_PER_TILE, 0, dummy>
          {
            // Init
            template<typename Tile>
            static __device__ __forceinline__ void Init(Tile *tile)
            {
            }

            // Inspect
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
            {
            }

            // ExpandByCta
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
            {
            }

            // ExpandByWarp
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
            {
            }

            // ExpandByScan
            template<typename Cta, typename Tile>
            static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
            {
            }
          };

          //---------------------------------------------------------------------
          // Interface
          //---------------------------------------------------------------------

          /**
           * Constructor
           */
          __device__ __forceinline__ Tile()
          {
            Iterate<0, 0>::Init(this);
          }

          /**
           * Inspect dequeued vertices, updating label if necessary and
           * obtaining edge-list details
           */
          template<typename Cta>
          __device__ __forceinline__ void Inspect(Cta *cta)
          {
            Iterate<0, 0>::Inspect(cta, this);
          }

          /**
           * Expands neighbor lists for valid vertices at CTA-expansion granularity
           */
          template<typename Cta>
          __device__ __forceinline__ void ExpandByCta(Cta *cta)
          {
            Iterate<0, 0>::ExpandByCta(cta, this);
          }

          /**
           * Expands neighbor lists for valid vertices a warp-expansion granularity
           */
          template<typename Cta>
          __device__ __forceinline__ void ExpandByWarp(Cta *cta)
          {
            Iterate<0, 0>::ExpandByWarp(cta, this);
          }

          /**
           * Expands neighbor lists by local scan rank
           */
          template<typename Cta>
          __device__ __forceinline__ void ExpandByScan(Cta *cta)
          {
            Iterate<0, 0>::ExpandByScan(cta, this);
          }
        }
        ;

        //---------------------------------------------------------------------
        // Methods
        //---------------------------------------------------------------------

        /**
         * Constructor
         */
        __device__ __forceinline__ Cta(int iteration,
            SizeT num_elements,
            VertexId queue_index,
            int num_gpus,
            int selector,
            int previous_frontier_size,
            int *deviceMappedValueEdge,
            SmemStorage &smem_storage,
            VertexId *d_in,
            VertexId *d_out,
            VertexId *d_predecessor_out,
            VertexType &vertex_list,
            EdgeType &edge_list,
            VertexId *d_edgeCSC_indices,
            char* d_changed,
            int* d_active_flags,
            VertexId *d_column_indices,
            SizeT *d_row_offsets,
            util::CtaWorkProgress &work_progress,
            SizeT max_edge_frontier) :
            iteration(iteration), num_elements(num_elements),
                queue_index(queue_index), num_gpus(num_gpus), selector(selector), previous_frontier_size(previous_frontier_size), deviceMappedValueEdge(deviceMappedValueEdge), smem_storage(
                    smem_storage), raking_soa_details(
                    typename RakingSoaDetails::GridStorageSoa(smem_storage.coarse_raking_elements, smem_storage.fine_raking_elements),
                    typename RakingSoaDetails::WarpscanSoa(smem_storage.state.coarse_warpscan, smem_storage.state.fine_warpscan),
                    TileTuple(0, 0)),
                d_in(d_in), d_out(d_out), d_predecessor_out(
                    d_predecessor_out), vertex_list(vertex_list), edge_list(edge_list), d_edgeCSC_indices(d_edgeCSC_indices), d_changed(d_changed), d_active_flags(d_active_flags), d_column_indices(
                    d_column_indices), d_row_offsets(
                    d_row_offsets), work_progress(
                    work_progress), max_edge_frontier(
                    max_edge_frontier)
        {
          if (threadIdx.x == 0)
          {
            smem_storage.state.cta_comm = KernelPolicy::THREADS;		// invalid
            smem_storage.state.overflowed = false;		// valid
          }
        }

        /**
         * Process a single tile
         */
        __device__ __forceinline__ void ProcessTile(SizeT cta_offset, SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
        {
//              if(blockIdx.x == 0 && threadIdx.x == 0) printf("In Expand ProcessTile\n");
          Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

          // Load tile
          util::io::LoadTile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE, KernelPolicy::THREADS, KernelPolicy::QUEUE_READ_MODIFIER, false>::LoadValid(tile.vertex_id, d_in,
              cta_offset, guarded_elements, (VertexId) -1);

          // Inspect dequeued vertices, updating label and obtaining
          // edge-list details
          tile.Inspect(this);
//              if(tile.vertex_id[0][0] != -1)
//            	  printf("before scan: bidx=%d, tidx=%d, tile.vertex_id[0][0]=%d, tile.coarse_row_rank[0][0] = %d, tile.fine_row_rank[0][0]=%d\n", blockIdx.x, threadIdx.x, tile.vertex_id[0][0], tile.coarse_row_rank[0][0], tile.fine_row_rank[0][0]);

          // Scan tile with carry update in raking threads
          SoaScanOp scan_op;
          TileTuple totals;
          util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(totals, raking_soa_details, RankSoa(tile.coarse_row_rank, tile.fine_row_rank), scan_op);

          SizeT coarse_count = totals.t0;
          tile.fine_count = totals.t1;
//              printf("In Expand ProcessTile, coarse_count=%d, tile.fine_count=%d\n",coarse_count, tile.fine_count);

          // Use a single atomic add to reserve room in the queue
          if (threadIdx.x == 0)
          {

            SizeT enqueue_amt = coarse_count + tile.fine_count;
//            SizeT enqueue_offset = work_progress.Enqueue(enqueue_amt, queue_index + 1);
            SizeT enqueue_offset = util::AtomicInt<SizeT>::Add(&deviceMappedValueEdge[selector], enqueue_amt);
//            if(enqueue_amt > 0)printf("blockIdx.x=%d, enqueue_amt=%d, deviceMappedValueEdge[0]=%d\n", blockIdx.x, enqueue_amt, deviceMappedValueEdge[0]);
//            if (blockIdx.x == 0)
//              printf("Expand: blockIdx.x=%d, enqueue_offset=%d, enqueue_amt=%d, deviceMappedValueEdge=%d\n", blockIdx.x, enqueue_offset, enqueue_amt, *deviceMappedValueEdge);

            smem_storage.state.coarse_enqueue_offset = enqueue_offset - previous_frontier_size;
            smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count - previous_frontier_size;

            // Check for queue overflow due to redundant expansion
            if (enqueue_offset + enqueue_amt >= max_edge_frontier)
            {
              if (blockIdx.x == 0)
                printf("Expand_dynamic_flag: queue size: %d, Frontier queue overflow.  Please increase queue-sizing factor.\n", enqueue_offset + enqueue_amt);
              smem_storage.state.overflowed = true;
              work_progress.SetOverflow<SizeT>();
            }
          }

          // Protect overflowed flag
          __syncthreads();

          // Quit if overflow
          if (smem_storage.state.overflowed)
          {
//                printf("EXPAND2: Overtflowed!!\n");
            util::ThreadExit();
          }

//              printf("processtile: bidx=%d, tidx=%d, coarse_count=%d, fine_count=%d\n", blockIdx.x, threadIdx.x, coarse_count, tile.fine_count);

          if (coarse_count > 0)
          {
            // Enqueue valid edge lists into outgoing queue
            tile.ExpandByCta(this);

            // Enqueue valid edge lists into outgoing queue
            tile.ExpandByWarp(this);
          }

          tile.progress = 0;
          while (tile.progress < tile.fine_count)
          {

            SizeT scratch_offset = tile.fine_row_rank[0][0] + tile.row_progress[0][0] - tile.progress;

            while ((tile.row_progress[0][0] < tile.row_length[0][0]) && (scratch_offset < SmemStorage::GATHER_ELEMENTS))
            {
              // Put gather offset into scratch space
              smem_storage.gather_offsets[scratch_offset] = tile.row_offset[0][0] + tile.row_progress[0][0];

              // Put dequeued vertex as the predecessor into scratch space
              smem_storage.gather_predecessors[scratch_offset] = tile.vertex_id[0][0];

              tile.row_progress[0][0]++;
              scratch_offset++;
            }

            __syncthreads();

            // Copy scratch space into queue
            int scratch_remainder = B40C_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);
//                if(blockIdx.x == 0 && threadIdx.x == 0)
//                    printf("GATHER_ELEMENTS=%d\n", SmemStorage::GATHER_ELEMENTS);
//
            for (int scratch_offset = threadIdx.x; scratch_offset < scratch_remainder; scratch_offset += KernelPolicy::THREADS)
            {
              // Gather a neighbor
              VertexId neighbor_id_tmp;
              VertexId row_id = smem_storage.gather_predecessors[scratch_offset];
//              typename Program::expand_vertex expand_vertex_functor;
//              bool changed = expand_vertex_functor(row_id, d_changed[row_id], vertex_list, edge_list); //might use optimization for unique load for a vertex
              bool changed = d_changed[row_id];

//                  util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, d_column_indices + smem_storage.gather_offsets[scratch_offset]);
              SizeT col_idx = smem_storage.gather_offsets[scratch_offset];
              neighbor_id_tmp = d_column_indices[col_idx];
//                  printf("row_id=%d, changed=%d, neighbor_id_tmp=%d\n", row_id, changed, neighbor_id_tmp);

              VertexId frontier;
              typename Program::MiscType misc_value;
              typename Program::expand_edge expand_edge_functor;

              int edge_id;
              if (d_edgeCSC_indices)
                edge_id = d_edgeCSC_indices[col_idx];
              else
                edge_id = col_idx;

              if (changed && atomicCAS(&d_active_flags[neighbor_id_tmp], 0, 1) == 0)
                expand_edge_functor(changed, iteration, row_id, neighbor_id_tmp, edge_id, vertex_list, edge_list, frontier, misc_value);
              else
                frontier = -1;

              d_out[smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset] = frontier;
            }

            tile.progress += SmemStorage::GATHER_ELEMENTS;

            __syncthreads();
          }
        }
      }
      ;

    } // namespace expand_atomic
  } // namespace vertex_centric
} // namespace GASengine

