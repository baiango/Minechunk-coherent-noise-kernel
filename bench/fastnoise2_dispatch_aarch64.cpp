#define FASTSIMD_MAX_FEATURE_SET AARCH64
#define FASTSIMD_LIBRARY_NAME FastSIMD_FastNoise

#include <FastSIMD/Utility/ArchDetect.h>

#if FASTSIMD_CURRENT_ARCH_IS( ARM )
#include <FastSIMD/FastSIMD_FastNoise_config.h>

#include <DispatchClassImpl.h>
#include <FastNoise/FastSIMD_Build.inl>
#endif
