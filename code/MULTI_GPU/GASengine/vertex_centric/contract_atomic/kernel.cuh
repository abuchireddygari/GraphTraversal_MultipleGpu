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

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <GASengine/vertex_centric/contract_atomic/cta.cuh>

using namespace b40c;
using namespace graph;

namespace GASengine
{
  namespace vertex_centric
  {
    namespace contract_atomic
    {

      /**
       * Contraction pass (non-workstealing)
       */
      template<typename KernelPolicy, typename Program, bool WORK_STEALING>
      struct SweepPass
      {

        static __device__ __forceinline__ void Invoke(typename KernelPolicy::VertexId &iteration,
                                                      typename Program::VertexId &queue_index,
                                                      typename Program::VertexId &steal_index,
                                                      int &num_gpus,
                                                      int &selector,
                                                      int *&d_frontier_size,
                                                      typename Program::VertexId *&d_edge_frontier,
                                                      typename Program::VertexId *&d_vertex_frontier,
                                                      typename Program::VertexId *&d_predecessor,
                                                      typename Program::GatherType *&m_gatherTmp,
                                                      typename Program::VertexType &vertex_list,
                                                      typename Program::EdgeType &edge_list,
                                                      char *&d_bitmap_visited,
                                                      typename KernelPolicy::VisitedMask *&d_visited_mask,
                                                      util::CtaWorkProgress &work_progress, util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition, typename KernelPolicy::SizeT &max_vertex_frontier,
                                                      typename KernelPolicy::SmemStorage & smem_storage)
        {
          typedef Cta<KernelPolicy, Program> Cta;
          typedef typename KernelPolicy::SizeT SizeT;

          // Determine our threadblock's work range
          util::CtaWorkLimits<SizeT> work_limits;
          work_decomposition.template GetCtaWorkLimits<KernelPolicy::LOG_TILE_ELEMENTS, KernelPolicy::LOG_SCHEDULE_GRANULARITY > (work_limits);

          // Return if we have no work to do
          if (!work_limits.elements)
          {
            return;
          }

          // CTA processing abstraction
          Cta cta(iteration, queue_index, num_gpus, selector, d_frontier_size, smem_storage, d_edge_frontier, 
                  d_vertex_frontier, d_predecessor, m_gatherTmp, vertex_list, edge_list, 
                  d_bitmap_visited, d_visited_mask, work_progress,
                  max_vertex_frontier);

          // Process full tiles
          while (work_limits.offset < work_limits.guarded_offset)
          {

            cta.ProcessTile(work_limits.offset);
            work_limits.offset += KernelPolicy::TILE_ELEMENTS;
          }

          // Clean up last partial tile with guarded-i/o
          if (work_limits.guarded_elements)
          {
            cta.ProcessTile(work_limits.offset, work_limits.guarded_elements);
          }
        }

      };

      /**
       * Atomically steal work from a global work progress construct
       */
      template<typename SizeT, typename StealIndex>
      __device__ __forceinline__ SizeT StealWork(util::CtaWorkProgress &work_progress, int count, StealIndex steal_index)
      {
        __shared__ SizeT s_offset; // The offset at which this CTA performs tile processing, shared by all

        // Thread zero atomically steals work from the progress counter
        if (threadIdx.x == 0)
        {
          s_offset = work_progress.Steal<SizeT > (count, steal_index);
        }

        __syncthreads(); // Protect offset

        return s_offset;
      }

      /**
       * Contraction pass (workstealing)
       */
      template<typename KernelPolicy, typename Program>
      struct SweepPass<KernelPolicy, Program, true >
      {

        static __device__ __forceinline__ void Invoke(typename KernelPolicy::VertexId &iteration, typename KernelPolicy::VertexId &queue_index,
                                                      typename KernelPolicy::VertexId &steal_index,
                                                      int &num_gpus, typename KernelPolicy::VertexId *&d_edge_frontier, typename KernelPolicy::VertexId *&d_vertex_frontier,
                                                      typename KernelPolicy::VertexId *&d_predecessor,
                                                      typename KernelPolicy::VertexId *&d_labels, typename Program::MiscType *&d_preds, typename KernelPolicy::EValue *&d_sigmas,
                                                      typename KernelPolicy::VisitedMask *&d_visited_mask,
                                                      util::CtaWorkProgress &work_progress,
                                                      util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
                                                      typename KernelPolicy::SizeT &max_vertex_frontier,
                                                      typename KernelPolicy::SmemStorage & smem_storage)
        {
          typedef Cta<KernelPolicy, Program> Cta;
          typedef typename KernelPolicy::SizeT SizeT;

          // CTA processing abstraction
          Cta cta(iteration, queue_index, num_gpus, smem_storage, d_edge_frontier, d_vertex_frontier, d_predecessor, d_labels, d_preds, d_sigmas, d_visited_mask, work_progress,
                  max_vertex_frontier);

          // Total number of elements in full tiles
          SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelPolicy::TILE_ELEMENTS - 1));

          // Worksteal full tiles, if any
          SizeT offset;
          while ((offset = StealWork<SizeT > (work_progress, KernelPolicy::TILE_ELEMENTS, steal_index)) < unguarded_elements)
          {
            cta.ProcessTile(offset);
          }

          // Last CTA does any extra, guarded work (first tile seen)
          if (blockIdx.x == gridDim.x - 1)
          {
            SizeT guarded_elements = work_decomposition.num_elements - unguarded_elements;
            cta.ProcessTile(unguarded_elements, guarded_elements);
          }
        }
      };

      /******************************************************************************
       * Arch dispatch
       ******************************************************************************/

      /**
       * Not valid for this arch (default)
       */
      template<typename KernelPolicy, typename Program, bool VALID = (__B40C_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
      struct Dispatch
      {
        typedef typename KernelPolicy::VertexId VertexId;
        typedef typename KernelPolicy::EValue EValue;
        typedef typename KernelPolicy::SizeT SizeT;
        typedef typename KernelPolicy::VisitedMask VisitedMask;

        static __device__ __forceinline__ void Kernel(VertexId &src, VertexId &iteration, SizeT &num_elements, VertexId &queue_index, VertexId &steal_index, int &num_gpus, volatile int *&d_done,
                                                      VertexId *&d_edge_frontier, VertexId *&d_vertex_frontier, VertexId *&d_predecessor, VertexId *&d_labels, VertexId *&d_preds, EValue *&d_sigmas, int *&d_dists, int *&d_changed,
                                                      VisitedMask *&d_visited_mask, util::CtaWorkProgress &work_progress, SizeT &max_edge_frontier, SizeT &max_vertex_frontier, util::KernelRuntimeStats & kernel_stats)
        {
          // empty
        }
      };

      /**
       * Valid for this arch (policy matches compiler-inserted macro)
       */
      template<typename KernelPolicy, typename Program>
      struct Dispatch<KernelPolicy, Program, true >
      {
        typedef typename KernelPolicy::VertexId VertexId;
        typedef typename KernelPolicy::EValue EValue;
        typedef typename KernelPolicy::SizeT SizeT;
        typedef typename KernelPolicy::VisitedMask VisitedMask;
        typedef typename Program::VertexType VertexType;
        typedef typename Program::EdgeType EdgeType;
        typedef typename Program::MiscType MiscType;
        typedef typename Program::GatherType GatherType;

        static __device__ __forceinline__ void Kernel(VertexId &src,
                                                      VertexId &iteration,
                                                      VertexId &queue_index,
                                                      VertexId &steal_index,
                                                      int &num_gpus,
                                                      int &selector,
                                                      int* &d_frontier_size,
                                                      int* &d_edge_frontier_size,
                                                      volatile int *&d_done,
                                                      VertexId *&d_edge_frontier,
                                                      VertexId *&d_vertex_frontier,
                                                      MiscType *&d_predecessor,
                                                      GatherType *&m_gatherTmp,
                                                      VertexType &vertex_list,
                                                      EdgeType &edge_list,
                                                      char *&d_changed,
                                                      char *&d_bitmap_visited,
                                                      VisitedMask *&d_visited_mask,
                                                      util::CtaWorkProgress &work_progress,
                                                      SizeT &max_edge_frontier,
                                                      SizeT &max_vertex_frontier,
                                                      util::KernelRuntimeStats & kernel_stats)
        {

          // Shared storage for the kernel
          __shared__ typename KernelPolicy::SmemStorage smem_storage;

          //          if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0))
          //          {
          //            kernel_stats.MarkStart();
          //          }

          //          printf("iteration=%d\n", iteration);
          //          if (iteration == 0)
          //          {
          //
          //            if (threadIdx.x < util::CtaWorkProgress::COUNTERS)
          //            {
          //
          //              // Reset all counters
          //              work_progress.template Reset<SizeT>();
          //
          //              // Determine work decomposition for first iteration
          //              if (threadIdx.x == 0)
          //              {
          //
          ////                    SizeT num_elements = 0;
          ////                    if (src != -1)
          ////                    {
          ////
          ////                      num_elements = 1;
          ////
          ////                      // We'll be the only block with active work this iteration.
          ////                      // Enqueue the source for us to subsequently process.
          ////                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(src, d_edge_frontier);
          ////
          ////                      // Enqueue predecessor of source
          ////                      typename KernelPolicy::VertexId predecessor = 100000000;
          ////                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(predecessor, d_predecessor);
          ////
          ////                      int init_dist = 0;
          ////                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(init_dist, vertex_list.d_dists + src);
          ////                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(init_dist, vertex_list.d_dists_out + src);
          ////
          ////                      int init_changed = 1;
          ////                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(init_changed, d_changed + src);
          ////                    }
          //
          //                // Initialize work decomposition in smem
          //                smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(num_elements, gridDim.x);
          //              }
          //            }
          //
          //            // Barrier to protect work decomposition
          //            __syncthreads();
          //
          //            // Don't do workstealing this iteration because without a
          //            // global barrier after queue-reset, the queue may be inconsistent
          //            // across CTAs
          //            SweepPass<KernelPolicy, Program, false>::Invoke(iteration, queue_index, steal_index, num_gpus, d_edge_frontier, d_vertex_frontier, d_predecessor,
          //                vertex_list, edge_list,
          //                d_labels, d_preds, d_sigmas, d_visited_mask,
          //                work_progress, smem_storage.state.work_decomposition, max_vertex_frontier, smem_storage);
          //
          //          }
          //          else

          // Determine work decomposition
          if (threadIdx.x == 0)
          {

            // Obtain problem size
            //              if (KernelPolicy::DEQUEUE_PROBLEM_SIZE)
            //              {
            //                num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
            //              }

            int num_elements = d_edge_frontier_size[selector];

            if (blockIdx.x == 0)
              d_frontier_size[selector] = 0;

            //            if(blockIdx.x == 0)
            //              printf("contract: num_elements=%d\n", num_elements);
            // Check if we previously overflowed
            if (num_elements >= max_edge_frontier)
            {
              num_elements = 0;
            }

            // Signal to host that we're done
            if ((num_elements == 0) || (KernelPolicy::SATURATION_QUIT && (num_elements <= gridDim.x * KernelPolicy::SATURATION_QUIT)))
            {
              if (d_done) d_done[0] = num_elements;
            }

            // Initialize work decomposition in smem
            smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY > (num_elements, gridDim.x);

            // Reset our next outgoing queue counter to zero
            //              work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

            // Reset our next workstealing counter to zero
            //              work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

          }

          // Barrier to protect work decomposition
          __syncthreads();

          SweepPass<KernelPolicy, Program, KernelPolicy::WORK_STEALING>::Invoke(iteration, queue_index, steal_index, num_gpus, selector, d_frontier_size, d_edge_frontier, d_vertex_frontier,
                                                                                d_predecessor, m_gatherTmp, vertex_list, edge_list, d_bitmap_visited, d_visited_mask, work_progress, smem_storage.state.work_decomposition, max_vertex_frontier, smem_storage);

        }

        //        if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0))
        //        {
        //          kernel_stats.MarkStop();
        //          kernel_stats.Flush();
        //        }

      };

      /******************************************************************************
       * Contraction Kernel Entrypoint
       ******************************************************************************/

      /**
       * Contraction kernel entry point
       */
      template<typename KernelPolicy, typename Program>
      __launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
      __global__
      void Kernel(typename KernelPolicy::VertexId src, // Source vertex (may be -1 if iteration != 0)
                  typename KernelPolicy::VertexId iteration, // Current BFS iteration
                  typename KernelPolicy::VertexId queue_index, // Current frontier queue counter index
                  typename KernelPolicy::VertexId steal_index, // Current workstealing counter index
                  int num_gpus, // Number of GPUs
                  int selector,
                  int* d_frontier_size,
                  int* d_edge_frontier_size,
                  volatile int *d_done, // Flag to set when we detect incoming edge frontier is empty
                  typename KernelPolicy::VertexId *d_edge_frontier, // Incoming edge frontier
                  typename KernelPolicy::VertexId *d_vertex_frontier, // Outgoing vertex frontier
                  typename Program::MiscType *d_predecessor, // Incoming predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
                  typename Program::GatherType* m_gatherTmp,
                  typename Program::VertexType vertex_list, //
                  typename Program::EdgeType edge_list, //
                  char *d_changed, //changed flag
                  char *d_bitmap_visited,
                  typename KernelPolicy::VisitedMask *d_visited_mask, // Mask for detecting visited status
                  util::CtaWorkProgress work_progress, // Atomic workstealing and queueing counters
                  typename KernelPolicy::SizeT max_edge_frontier, // Maximum number of elements we can place into the outgoing edge frontier
                  typename KernelPolicy::SizeT max_vertex_frontier, // Maximum number of elements we can place into the outgoing vertex frontier
                  util::KernelRuntimeStats kernel_stats) // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
      {
        Dispatch<KernelPolicy, Program>::Kernel(src, iteration, queue_index, steal_index, num_gpus, selector, d_frontier_size, d_edge_frontier_size, d_done, d_edge_frontier, d_vertex_frontier,
                                                d_predecessor, m_gatherTmp, vertex_list, edge_list,
                                                d_changed, d_bitmap_visited,
                                                d_visited_mask, work_progress, max_edge_frontier, max_vertex_frontier, kernel_stats);
      }

    } // namespace contract_atomic
  } // namespace vertex_centric
} // namespace GASengine

