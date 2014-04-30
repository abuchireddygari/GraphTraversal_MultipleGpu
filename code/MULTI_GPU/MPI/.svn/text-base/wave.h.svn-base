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

#include "mpi.h"
#include "kernel.cuh"
#include <GASengine/statistics.h>
#ifndef WAVE_H_
#define WAVE_H_
using namespace std;
using namespace MPI;
using namespace mpikernel;

class wave
//frontier contraction in a 2-d partitioned graph
{
public:
  int pi; //row
  int pj; //column
  int p;
  int n;
  MPI_Group orig_group, new_row_group, new_col_group;
  MPI_Comm new_row_comm, new_col_comm;
  int new_row_rank, new_col_rank;
  double init_time, propagate_time, broadcast_time;
  Statistics* stats;
public:

  wave(int l_pi, int l_pj, int l_p, int l_n, Statistics* l_stats)
  //l_pi is the x index
  //l_pj is the y index
  //l_p  is the number of partitions in 1d. usually, sqrt(number of processors)
  //l_n  is the size of the problem, number of vertices
  {
    double starttime, endtime;
    starttime = MPI_Wtime();
    pi = l_pi;
    pj = l_pj;
    p = l_p;
    n = l_n;
    stats = l_stats;

    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    //build original ranks for the processors

    int row_indices[p], col_indices[p + 1];
    for (int i = 0; i < p; i++)
      row_indices[i] = pi * p + i;
    /*		for(int i=0;i<=pi-1;i++)
     row_indices[i+p] = i*p+pi;
     for(int i=pi+1;i<p;i++)
     row_indices[i+p-1] = i*p+pi;
     */for (int i = 0; i < p; i++)
      col_indices[i] = i * p + pj;
    /*              for(int i=0;i<=pj-1;i++)
     col_indices[i] = i*p+pj;
     for(int i=pj+1;i<p;i++)
     col_indices[i-1] = i*p+pj;
     col_indices[p-1] = pj*p+p-1;
     */
    MPI_Group_incl(orig_group, p, row_indices, &new_row_group);
    MPI_Group_incl(orig_group, p, col_indices, &new_col_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_row_group, &new_row_comm);
    MPI_Comm_create(MPI_COMM_WORLD, new_col_group, &new_col_comm);
    MPI_Group_rank(new_row_group, &new_row_rank);
    MPI_Group_rank(new_col_group, &new_col_rank);
    endtime = MPI_Wtime();
    init_time = endtime - starttime;
    propagate_time = 0;
    broadcast_time = 0;
  }

  void propogate(char* out_d, char* assigned_d, char* prefix_d)
  //wave propogation, in sequential from top to bottom of the column
  {
    double starttime, endtime;
    starttime = MPI_Wtime();
    unsigned int mesg_size = ceil(n / 8.0);
    int myid = pi * p + pj;
    //int lastid = pi*p+p-1;
    int numthreads = 512;
    int byte_size = (n + 8 - 1) / 8;
    int numblocks = min(512, (byte_size + numthreads - 1) / numthreads);

    MPI_Request request[2];
    MPI_Status status[2];
    if (p > 1)
    {
      //if first one in the column, initiate the wave propogation
      if (pj == 0)
      {
        char *out_h = (char*)malloc(mesg_size);
        cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);

        MPI_Isend(out_h, mesg_size, MPI_CHAR, myid + 1, pi, MPI_COMM_WORLD, &request[1]);
        MPI_Wait(&request[1], &status[1]);
        free(out_h);
      }
        //else if not the last one, receive bitmap from top, process and send to next one
      else if (pj != p - 1)
      {
        char *prefix_h = (char*)malloc(mesg_size);
        MPI_Irecv(prefix_h, mesg_size, MPI_CHAR, myid - 1, pi, MPI_COMM_WORLD, &request[0]);
        MPI_Wait(&request[0], &status[0]);

        cudaMemcpy(prefix_d, prefix_h, mesg_size, cudaMemcpyHostToDevice);
        mpikernel::bitsubstract << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, assigned_d);
        cudaDeviceSynchronize();
        mpikernel::bitunion << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, out_d);
        char *out_h = (char*)malloc(mesg_size);
        cudaDeviceSynchronize();
        cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);

        MPI_Isend(out_h, mesg_size, MPI_CHAR, myid + 1, pi, MPI_COMM_WORLD, &request[1]);
        free(prefix_h);

        MPI_Wait(&request[1], &status[1]);
        free(out_h);
      }
        //else receive from the previous and then broadcast to the broadcast group
      else
      {
        char *prefix_h = (char*)malloc(mesg_size);
        MPI_Irecv(prefix_h, mesg_size, MPI_CHAR, myid - 1, pi, MPI_COMM_WORLD, &request[0]);
        MPI_Wait(&request[0], &status[0]);
        cudaMemcpy(prefix_d, prefix_h, mesg_size, cudaMemcpyHostToDevice);
        mpikernel::bitsubstract << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, assigned_d);
        cudaDeviceSynchronize();
        mpikernel::bitunion << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, out_d);
        cudaDeviceSynchronize();
      }
    }

    endtime = MPI_Wtime();
    propagate_time += endtime - starttime;
  }

//Version that does not support GPUDirect 
  void reduce_frontier_CPU(char* out_d, char* in_d)
  {
    double starttime, endtime;
    starttime = MPI_Wtime();
    unsigned int mesg_size = ceil(n / (8.0));
    char *out_h = (char*)malloc(mesg_size);
    char *out_h2 = (char*)malloc(mesg_size);
    char *in_h = (char*)malloc(mesg_size);
    cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    MPI_Allreduce(out_h, out_h2, mesg_size, MPI_BYTE, MPI_BOR, new_row_comm);

    cudaMemcpy(out_d, out_h2, mesg_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    endtime = MPI_Wtime();
    propagate_time += endtime - starttime;

    starttime = MPI_Wtime();
    if (pi == pj)
      memcpy(in_h, out_h2, mesg_size);

    MPI_Bcast(in_h, mesg_size, MPI_CHAR, pj, new_col_comm);
    cudaMemcpy(in_d, in_h, mesg_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(in_h);
    free(out_h);
    endtime = MPI_Wtime();
    broadcast_time += endtime - starttime;
  }

//version that supports GPUDirect
  void reduce_frontier_GDR(char* out_d, char* in_d)
  {
    double starttime, endtime;
    starttime = MPI_Wtime();
    unsigned int mesg_size = ceil(n / (8.0));

    MPI_Allreduce(out_d, out_d, mesg_size, MPI_BYTE, MPI_BOR, new_row_comm);
    endtime = MPI_Wtime();

    propagate_time += endtime - starttime;

    starttime = MPI_Wtime();
    if (pi == pj)
	cudaMemcpy(in_d, out_d, mesg_size, cudaMemcpyDeviceToDevice);

    MPI_Bcast(in_d, mesg_size, MPI_CHAR, pj, new_col_comm);

    endtime = MPI_Wtime();
    broadcast_time += endtime - starttime;

  }


  void broadcast_new_frontier(char* out_d, char* in_d)
  {
    double starttime, endtime;
    starttime = MPI_Wtime();

    unsigned int mesg_size = ceil(n / (8.0));

    char *out_h = (char*)malloc(mesg_size);
    char *in_h = (char*)malloc(mesg_size);

    if (pj == p - 1)
      cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);

    MPI_Bcast(out_h, mesg_size, MPI_CHAR, p - 1, new_row_comm);
    cudaMemcpy(out_d, out_h, mesg_size, cudaMemcpyHostToDevice);

    if (pi == pj)
      memcpy(in_h, out_h, mesg_size);

    MPI_Bcast(in_h, mesg_size, MPI_CHAR, pj, new_col_comm);

    cudaMemcpy(out_d, out_h, mesg_size, cudaMemcpyHostToDevice);
    cudaMemcpy(in_d, in_h, mesg_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    free(in_h);
    free(out_h);
    endtime = MPI_Wtime();
    broadcast_time += endtime - starttime;
  }

};

#endif /* WAVE_H_ */
