/*
 Copyright (C) SYSTAP, LLC 2006-2014.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/*! \mainpage MapGraph

\section intro_sec Introduction

MapGraph is Massively Parallel Graph processing on GPUs. (Previously known as
"MPGraph").

- The MapGraph API makes it easy to develop high performance graph
analytics on GPUs. The API is based on the Gather-Apply-Scatter (GAS)
model as used in GraphLab. To deliver high performance computation and
efficiently utilize the high memory bandwidth of GPUs, MapGraph's CUDA
kernels use multiple sophisticated strategies, such as
vertex-degree-dependent dynamic parallelism granularity and frontier
compaction.

- New algorithms can be implemented in a few hours that fully exploit
the data-level parallelism of the GPU and offer throughput of up to 3
billion traversed edges per second on a single GPU.

- Partitioned graphs and Multi-GPU support will be in a future release.

- The MapGraph API also comes in a CPU-only version that is currently packaged 
and distributed with the <a href="http://sourceforge.net/projects/bigdata/">
bigdata open-source graph database</a>. GAS programs operate over the graph
data loaded into the database and are accessed via either a Java API or a
<a href="http://www.w3.org/TR/sparql11-federated-query/">SPARQL 1.1 Service Call
</a>. Packaging the GPU version inside bigdata will be in a future release.

MapGraph is under the <a
href="http://www.apache.org/licenses/LICENSE-2.0.html" > Apache 2
license</a>.  You can download MapGraph from <a
href="http://sourceforge.net/projects/mpgraph/" >
http://sourceforge.net/projects/mpgraph/ </a>. For the lastest version
of this documentation, see <a
href="http://www.systap.com/mapgraph/api/html/index.html">
http://www.systap.com/mapgraph/api/html/index.html </a>.  You can
subscribe to receive notice for future updates on the <a
href="http://sourceforge.net/projects/mpgraph/" >project home
page</a>.  For open source support, please ask a question on the <a
href="https://sourceforge.net/p/mpgraph/mailman/" >MapGraph mailing
lists</a> or file a <a
href="https://sourceforge.net/p/mpgraph/tickets/" >ticket</a>.  To
inquire about commercial support, please email us at
licenses@bigdata.com.  You can follow MapGraph and the bigdata graph
database platform at <a href="http://www.bigdata.com/blog"
>http://www.bigdata.com/blog</a>.

This work was (partially) funded by the DARPA XDATA program under AFRL
<tt>Contract #FA8750-13-C-0002</tt>.

\subsection performance MapGraph vs Many-Core CPUs

MapGraph is up to two orders of magnitude faster than parallel CPU
implementations on up 24 CPU cores and has performance comparable to a
state-of-the-art manually optimized GPU implementation.  For example,
the diagram below shows the speedups of MapGraph versus GraphLab for SSSP.

\image html MapGraphv2-vs-GraphLab-SSSP.jpg "MapGraph v2 Speedups (versus GraphLab, SSSP, N-core)"

For our GPU evaluations we used a NVIDIA c2075 (Fermi architecture), but
performance is similar on other NVIDIA cards.
The CPU platform was a machine containing a 3.33 GHz X5680 CPU chipset.
This is a dual-socket Westmere chipset that contains 12 physical cores
and 12 MB of cache. The machine contains 24 GB of 1333 MHz ECC memory.
The software environment is RedHat 6.2 Beta. CPU code was compiled with
gcc (GCC) 4.4.6 20110731 (Red Hat 4.4.6-3). The results were obtained
using the synchronous engine for GraphLab due to core faults with some
data sets when using the asynchronous engine.

\section api The MapGraph API

MapGraph is implemented as a set of templates following a design
pattern that is similar to the Gather-Apply-Scatter (GAS) API. GAS is
a vertex-centric API, similar to the API first popularized by Pregel.
The GAS API breaks down operations into the following phases:

- Gather  - reads data from the one-hop neighborhood of a vertex.
- Apply   - updates the vertex state based on the gather result.
- Scatter - pushes updates to the one-hop neighborhood of a vertex.

The GAS API has been extended in order to: (a) maximize parallelism;
(b) manage simultaneous discovery of duplicate vertices (this is not
an issue in multi-core CPU code); (c) provide appropriate memory
barriers (each kernel provides a memory barrier); (d) optimize memory
layout; and (e) allow "push" style scatter operators are similar to
"signal" with a message value to create a side-effect in GraphLab.

\subsection kernels MapGraph Kernels

MapGraph defines the following kernels and supports their invocation
from templated CUDA programs.  Each kernel may have one or more device
functions that it invokes.  User code (a) provides implementations of
those device functions to customize the behavior of the algorithm; and
(b) provides custom data structures for the vertices and links (see
below).

Gather Phase Kernels::

- gather: The gather kernel reads data from the one-hop neighborhood
  of each vertex in the frontier.

Apply Phase Kernels::

- apply: The apply kernel updates the state of each vertex in the
  frontier given the results of the most recent gather or scatter
  operation.

- post-apply: The post-apply kernel runs after all threads in the
  apply() function have had the opportunity to synchronize at a memory
  barrier.

Scatter Phase Kernels::

- expand: The expand kernel creates the new frontier.

- contract: The contract kernel eliminates duplicates in the frontier
  (this is the problem of simultaneous discovery).

\subsection data_structures MapGraph Data Structures

In order to write code to the MapGraph API, it helps to have a
high-level understanding of the data structures used to maintain the
frontier, the topology, and the user data associated with the vertices
and edges of the graph.

\subsubsection frontier Frontier Queues

MapGraph uses frontier queues to maintain a dense list of the active
vertices.  These queues are managed by the MapGraph kernels, but user
data may be allocated and accessed that is 1:1 with the frontier (see
below).  The frontier array dimensions are determined by the number of
vertices in the graph times the frontier queue size multiplier. The
frontier is in global memory, but is buffered in shared memory by some
kernels.

\image html MapGraph-Frontier.jpg "MapGraph Frontier Queues"

The frontier is populated by the expand() kernel.  The contract()
kernel may be used to eliminate some or all of the duplicates
depending on the strategy and the needs of the graph algorithm. There
are actually two two frontier queues - this is for double-buffering.

In addition to the frontier, you may allocate optional user data
arrays that are 1:1 with the active vertices in the frontier.  These
arrays provides an important scratch area for many calculations and
benefit from dense, coalesced access. The arrays are accessed from the
same kernels that operate on the vertex frontier.  For example, BFS
uses a scratch array to store the predecessor value.

\subsubsection frontier Graph Topology

The topology of the graph is modeled by a forward and reverse sparse
matrix and is constructed at runtime from the sparse matrix data file.
The digrams below illustrate the use of CSR and CSC data structures to
model the graph topology. However, these topology data structures are
not directly exposed to user algorithms and their internals may
change.  Users write device functions that are invoked from kernels
that process the topology using a variety of different strategies.
Users do not need to access or understand the internals of the
topology data structures to write graph algorithms.

\image html MapGraph-Topology.jpg "MapGraph Topology"

The forward topology index is currently a Compressed Sparse Row (CSR)
matrix that provides row based indexing into the graph.  This data
structure is not directly exposed to user algorithms.  CSR is used to
access (traverse) the out-edges of the graph.

The reverse topology index is currently a Compressed Sparse Column
(CSC) matrix that provides column based indexing into the graph.  This
data structure is not directly exposed to user algorithms.  CSC is
used to access (traverse) the in-edges of the graph. The CSC edgeId
array gives the index into the EdgeList arrays. The CSC data structure
is only maintained if the algorithm will read over the in-edges.

\subsubsection user-data User Data

User data is specific to a given algorithm.  It is laid out in a
structure of arrays format in order to maximize coalesced memory
access.  (User data can also be 1:1 with the frontier - see above.)
There are two basic user data structures: The vertex list and the edge
list.

\image html MapGraph-UserData.jpg "MapGraph User Data"

The VertexList is a structure of named arrays that provides data for
each vertex in the graph.  The index into each array is the vertexId.
See the VertexData structure in one of the existing algorithms for
examples.

The EdgeList is a structure of named arrays that provides data for
each edge in the graph.  The index into each array is the edgeId. The
CSR.colind[] and the CSC.edgeId[] are both 1:1 with the edge list
arrays. Again, see an EdgeData structure in one of the existing
algorithms for examples.

The vertex list and edge list are laid out in vertical stripes using a
Structures of Arrays pattern for optimal memory access patterns on the
GPU.  To add your own data, you add a field to the vertex data struct
or the edge data struct. That field will be an array that is 1:1 with
the vertex identifiers.  You will need to initialize your array.
MapGraph will provide you with access to your data from within the
appropriate device functions.

\subsection write_your_own Writing your own MapGraph algorithm

MapGraph is based on templates.  This means that there is no interface
or super class from which you can derive your code.  Instead, you need
to start with one of the existing implementations that uses the
MapGraph template "pattern". You then need to review and modify the
function that initializes the user data structures (the vertex list
and the edge list) and the device functions that implement the user
code for the Gather, Apply, and Scatter primitives.  You can also
define functions that will extract the results from the GPU.

*/
