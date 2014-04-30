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
#include <stdlib.h>

template <class T> 
inline T getValue(const char* name);

template <> 
inline int getValue<int>(const char* name) {
  return atoi(name);
}

template <> 
inline bool getValue<bool>(const char* name) {
  if(strcmp(name, "true"))
		  return true;
  else if(strcmp(name, "false"))
	  return false;
  else
  {
	  std::cout << "Bool value error, use default value false" << std::endl;
	  return false;
  }
}

template <>
inline float getValue<float>(const char* name) {
  return atof(name);
}

template <> 
inline double getValue<double>(const char* name) {
  return atof(name);
}
