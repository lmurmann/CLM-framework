#ifndef SIMPLECLM_H
#define SIMPLECLM_H

#include <stdint.h>

#ifdef BUILD_SIMPLECLM
#define CLMAPI __declspec( dllexport ) 
#else
#define CLMAPI __declspec( dllimport ) 
#endif

struct SimpleCLM;
// NOT GOOD
CLMAPI int SimpleCLM_run (int argc, char **argv);
CLMAPI SimpleCLM* SimpleCLM_create();
CLMAPI int SimpleCLM_initModel(SimpleCLM* sclm, int argc, char** argv);
CLMAPI void SimpleCLM_initIntrinsics(SimpleCLM* sclm, int videow, int videoh);
CLMAPI void SimpleCLM_setFocalLength(SimpleCLM* sclm, float cx, float cy, float fx, float fy);
CLMAPI void SimpleCLM_processFrame(SimpleCLM* sclm, uint8_t* data, int w, int h);
CLMAPI void SimpleCLM_visualize(SimpleCLM* sclm, uint8_t* data, int w, int h);
CLMAPI void SimpleCLM_getPose(SimpleCLM* sclm, float* pos, float* rot);
CLMAPI void SimpleCLM_free(SimpleCLM*);
#endif
