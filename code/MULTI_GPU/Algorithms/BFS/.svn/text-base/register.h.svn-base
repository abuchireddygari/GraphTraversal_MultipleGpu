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

#include <config.h>
#include <limits.h>

//register the parameters from command line or configuration file
inline void registerParameters() {
//	Config::registerParameter<std::string>("source_file_name","the file of starting vertices", std::string(""));
  Config::registerParameter<int>("src","Starting vertex (default 1)", 1); // starting vertex
  Config::registerParameter<int>("verbose","Print out frontier size in each iteration (default 0)", 0);//print more infor
  Config::registerParameter<int>("origin","The origin (0 or 1) for the starting vertices (default 1)", 1); //vertex indices origin
  Config::registerParameter<int>("directed","The graph is directed (default 1)", 1); //whether the graph is directed or not
  Config::registerParameter<int>("device","The device to use (default 0)", 0); // the device number
  Config::registerParameter<int>("iter_num","The number of iterations to perform (default INT_MAX)", INT_MAX);
  Config::registerParameter<int>("num_src","The number of starting vertices when random sources is specified (default 1)", 1);
  Config::registerParameter<int>("run_CPU","Run CPU implementation for testing (default 0)", 0);
//  Config::registerParameter<int>("num_vertices","Number of vertices for the random graph (default 10)", 10);
//  Config::registerParameter<int>("num_edges","Number of edges for the random graph (default 100)", 100);
  Config::registerParameter<int>("with_value","Whether to load edge values from market file (default 0)", 0); // the device number
  Config::registerParameter<double>("max_queue_sizing","The frontier queue size is this value times the number of vertices in the graph (default 1.5)", 1.5); //frontier queue size
  Config::registerParameter<int>("threshold","When frontier size is larger than threshold, two-phase strategy is used otherwise dynamic scheduling it used (default 10000)", 10000);
}
