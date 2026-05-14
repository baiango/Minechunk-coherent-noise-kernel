#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <FastNoise/FastNoise.h>
#include <FastSIMD/ToolSet.h>

namespace
{
constexpr float kGridStep = 0.05f;
constexpr int kSeed = 1337;

struct RunResult
{
    std::uint64_t elapsed_ns = std::numeric_limits<std::uint64_t>::max();
    double checksum = 0.0;
};

bool gridSampleCount( std::size_t grid_size, std::size_t& samples )
{
    if( grid_size == 0 )
    {
        return false;
    }
    if( grid_size > std::numeric_limits<std::size_t>::max() / grid_size )
    {
        return false;
    }
    const std::size_t plane = grid_size * grid_size;
    if( grid_size > std::numeric_limits<std::size_t>::max() / plane )
    {
        return false;
    }
    samples = plane * grid_size;
    return true;
}

double checksumSamples( const float* values, std::size_t count )
{
    double sum = 0.0;
    const std::size_t stride = std::max<std::size_t>( 1, count / 64 );

    for( std::size_t index = 0; index < count; index += stride )
    {
        sum += values[index];
    }

    return sum;
}

double medianElapsedNs( std::vector<std::uint64_t>& timings )
{
    std::sort( timings.begin(), timings.end() );
    const std::size_t midpoint = timings.size() / 2;
    if( timings.size() % 2 != 0 )
    {
        return static_cast<double>( timings[midpoint] );
    }
    return ( static_cast<double>( timings[midpoint - 1] ) + static_cast<double>( timings[midpoint] ) ) * 0.5;
}

RunResult runFastNoise2Passes(
    const FastNoise::SmartNode<>& generator,
    float* output,
    std::size_t grid_size,
    std::size_t samples,
    std::size_t passes )
{
    const int grid_side = static_cast<int>( grid_size );
    int seed = kSeed;
    const auto start = std::chrono::steady_clock::now();

    for( std::size_t pass = 0; pass < passes; ++pass )
    {
        const float offset = static_cast<float>( pass ) * 0.125f;
        generator->GenUniformGrid3D(
            output,
            offset,
            -offset * 0.5f,
            offset * 0.25f,
            grid_side,
            grid_side,
            grid_side,
            kGridStep,
            kGridStep,
            kGridStep,
            seed++ );
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();

    return RunResult{
        static_cast<std::uint64_t>( elapsed_ns ),
        checksumSamples( output, samples ),
    };
}
}

extern "C"
{
struct FastNoise2SimplexBenchResult
{
    double median_elapsed_ns;
    std::size_t samples;
    double checksum;
    const char* active_feature;
};

std::uint8_t minechunk_bench_fastnoise2_simplex_uniform_grid3d(
    float* output,
    std::size_t output_len,
    std::size_t grid_size,
    std::size_t passes,
    std::size_t repeats,
    FastNoise2SimplexBenchResult* result )
{
    std::size_t samples = 0;
    if( output == nullptr || result == nullptr || !gridSampleCount( grid_size, samples ) ||
        grid_size > static_cast<std::size_t>( std::numeric_limits<int>::max() ) ||
        output_len < samples || passes == 0 || repeats == 0 )
    {
        return 0;
    }

    FastNoise::SmartNode<FastNoise::Simplex> simplex =
        FastNoise::New<FastNoise::Simplex>( FastSIMD::FeatureSet::Max );
    if( !simplex )
    {
        return 0;
    }

    simplex->SetScale( 20.0f );
    FastNoise::SmartNode<> generator = simplex;

    double checksum = 0.0;
    std::vector<std::uint64_t> timings;
    timings.reserve( repeats );
    for( std::size_t repeat = 0; repeat < repeats; ++repeat )
    {
        const RunResult run = runFastNoise2Passes( generator, output, grid_size, samples, passes );
        timings.push_back( run.elapsed_ns );
        checksum = run.checksum;
    }

    result->median_elapsed_ns = medianElapsedNs( timings );
    result->samples = samples * passes;
    result->checksum = checksum;
    result->active_feature = FastSIMD::GetFeatureSetString( generator->GetActiveFeatureSet() );
    return 1;
}
}
