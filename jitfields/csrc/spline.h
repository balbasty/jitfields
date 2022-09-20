#ifndef JF_SPLINE
#define JF_SPLINE

// This file contains static functions for handling (0-7 order)
// spline weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . spline::weight     -> node weight based on distance
// . spline::fastweight -> same, assuming x lies in support
// . spline::grad       -> weight derivative // oriented distance
// . spline::fastgrad   -> same, assuming x lies in support
// . spline::hess       -> weight 2nd derivative // oriented distance
// . spline::fasthess   -> same, assuming x lies in support
// . spline::bounds     -> min/max nodes

// NOTE:
// 1st derivatives used to be implemented with a recursive call, e.g.:
// scalar_t grad2(scalar_t x) {
//   if (x < 0) return -grad2(-x);
//   ...
// }
// However, this prevents nvcc to staticallly determine the stack size
// and leads to memory errors (because the allocated stack is too small).
// I now use a slighlty less compact implementation that gets rid of
// recursive calls.

// TODO:
// . second order derivatives [5/6/7]
// ? other types of basis functions (gauss, sinc)

#include "cuda_switch.h"

namespace jf {
namespace spline {

enum class type : char {
    Nearest,
    Linear,
    Quadratic,
    Cubic,
    FourthOrder,
    FifthOrder,
    SixthOrder,
    SeventhOrder
};

namespace _spline {

  // --- order 0 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight0(scalar_t x) {
    x = fabs(x);
    return x < 0.5 ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight0(scalar_t x) {
    x = fabs(x);
    return static_cast<scalar_t>(1);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t hess0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fasthess0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds0(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(round(x));
    upp = low;
  }

  // --- order 1 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight1(scalar_t x) {
    x = fabs(x);
    return x < 1 ? static_cast<scalar_t>(1) - x : static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight1(scalar_t x) {
    return static_cast<scalar_t>(1) - fabs(x);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad1(scalar_t x) {
    if (fabs(x) >= 1) return static_cast<scalar_t>(0);
    return fastgrad1(x);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad1(scalar_t x) {
    return x < static_cast<scalar_t>(0) ? static_cast<scalar_t>(1)
                                        : static_cast<scalar_t>(-1);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t hess1(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fasthess1(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds1(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x));
    upp = low + 1;
  }

  // --- order 2 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight2(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return 0.75 - x * x;
    }
    else if ( x < 1.5 )
    {
      x = 1.5 - x;
      return 0.5 * x * x;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight2(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return 0.75 - x * x;
    }
    else
    {
      x = 1.5 - x;
      return 0.5 * x * x;
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad2(scalar_t x) {
    bool neg = x < 0;
    if ( x < 0.5 )
    {
      x = -2. * x;
    }
    else if ( x < 1.5 )
    {
      x = x - 1.5;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad2(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 0.5 )
    {
      x = -2. * x;
    }
    else
    {
      x = x - 1.5;
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t hess2(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return static_cast<scalar_t>(-2.);
    }
    else if ( x < 1.5 )
    {
      return static_cast<scalar_t>(1.);
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fasthess2(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return static_cast<scalar_t>(-2.);
    }
    else
    {
      return static_cast<scalar_t>(1.);
    }
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds2(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-.5));
    upp = low + 2;
  }

  // --- order 3 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight3(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      return ( x * x * (x - 2.) * 3. + 4. ) / 6.;
    }
    else if ( x < 2. )
    {
      x = 2. - x;
      return ( x * x * x ) / 6.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight3(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      return ( x * x * (x - 2.) * 3. + 4. ) / 6.;
    }
    else
    {
      x = 2. - x;
      return ( x * x * x ) / 6.;
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad3(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      x = x * ( x * 1.5 - 2. );
    }
    else if ( x < 2. )
    {
      x = 2. - x;
      x = - ( x * x ) * 0.5;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad3(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      x = x * ( x * 1.5 - 2. );
    }
    else
    {
      x = 2. - x;
      x = - ( x * x ) * 0.5;
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t hess3(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      return x * 3. - 2.;
    }
    else if ( x < 2. )
    {
      return 2. - x;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fasthess3(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      return x * 3. - 2.;
    }
    else
    {
      return 2. - x;
    }
  }


  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds3(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-1.));
    upp = low + 3;
  }

  // --- order 4 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight4(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * 0.25 - 0.625 ) + 115. / 192.;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( 5. - x ) / 6. - 1.25 ) + 5. / 24. ) + 55. / 96.;
    }
    else if ( x < 2.5 )
    {
      x -= 2.5;
      x *= x;
      return ( x * x ) / 24.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight4(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * 0.25 - 0.625 ) + 115. / 192.;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( 5. - x ) / 6. - 1.25 ) + 5. / 24. ) + 55. / 96.;
    }
    else
    {
      x -= 2.5;
      x *= x;
      return ( x * x ) / 24.;
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad4(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 0.5 )
    {
      x = x * ( x * x - 1.25 );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
    }
    else if ( x < 2.5 )
    {
      x = x * 2. - 5.;
      x = ( x * x * x ) / 48.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad4(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 0.5 )
    {
      x = x * ( x * x - 1.25 );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
    }
    else
    {
      x = x * 2. - 5.;
      x = ( x * x * x ) / 48.;
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t hess4(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return ( x * x ) * 3. - 1.25;
    }
    else if ( x < 1.5 )
    {
      return  x * ( x * ( -2. ) + 5. ) - 2.5;
    }
    else if ( x < 2.5 )
    {
      x = x * 2. - 5.;
      return ( x * x ) / 8.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fasthess4(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return ( x * x ) * 3. - 1.25;
    }
    else if ( x < 1.5 )
    {
      return  x * ( x * ( -2. ) + 5. ) - 2.5;
    }
    else
    {
      x = x * 2. - 5.;
      return ( x * x ) / 8.;
    }
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds4(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-1.5));
    upp = low + 4;
  }

  // --- order 5 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight5(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( 0.25 - x * ( 1. / 12. ) ) - 0.5 ) + 0.55;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( 1. / 24. ) - 0.375 ) + 1.25 ) -
             1.75 ) + 0.625 ) + 0.425;
    }
    else if ( x < 3. )
    {
      scalar_t f = 3. - x;
      x = f * f;
      return f * x * x * ( 1. / 120. );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight5(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( 0.25 - x * ( 1. / 12. ) ) - 0.5 ) + 0.55;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( 1. / 24. ) - 0.375 ) + 1.25 ) -
             1.75 ) + 0.625 ) + 0.425;
    }
    else
    {
      scalar_t f = 3. - x;
      x = f * f;
      return f * x * x * ( 1. / 120. );
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad5(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      x = x * ( x * ( x * ( x * ( -5. / 12. ) + 1. ) ) - 1. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( 5. / 24. ) - 1.5 ) + 3.75 ) - 3.5 ) + 0.625;
    }
    else if ( x < 3. )
    {
      x -= 3.;
      x *= x;
      x = - ( x * x ) / 24.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad5(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      x = x * ( x * ( x * ( x * ( -5. / 12. ) + 1. ) ) - 1. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( 5. / 24. ) - 1.5 ) + 3.75 ) - 3.5 ) + 0.625;
    }
    else
    {
      x -= 3.;
      x *= x;
      x = - ( x * x ) / 24.;
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds5(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-2.));
    upp = low + 5;
  }

  // --- order 6 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight6(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * ( 7. / 48. - x * ( 1. / 36. ) ) - 77. / 192. ) +
             5887. / 11520.0;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 1. / 48. ) - 7. / 48. ) +
             0.328125 ) - 35. / 288. ) - 91. / 256. ) - 7. / 768. ) +
             7861. / 15360.0;
    }
    else if ( x < 2.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( 7. / 60. - x * ( 1. / 120. ) ) -
             0.65625 ) + 133. / 72. ) - 2.5703125 ) + 1267. / 960. ) +
             1379. / 7680.0;
    }
    else if ( x < 3.5 )
    {
      x -= 3.5;
      x *= x * x;
      return x * x * ( 1. / 720. );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight6(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * ( 7. / 48. - x * ( 1. / 36. ) ) - 77. / 192. ) +
             5887. / 11520.0;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 1. / 48. ) - 7. / 48. ) +
             0.328125 ) - 35. / 288. ) - 91. / 256. ) - 7. / 768. ) +
             7861. / 15360.0;
    }
    else if ( x < 2.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( 7. / 60. - x * ( 1. / 120. ) ) -
             0.65625 ) + 133. / 72. ) - 2.5703125 ) + 1267. / 960. ) +
             1379. / 7680.0;
    }
    else
    {
      x -= 3.5;
      x *= x * x;
      return x * x * ( 1. / 720. );
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad6(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < .5 )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 * ( 7. / 12. ) - ( x2 * x2 ) / 6.- 77./96. );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( x * ( x * 0.125 - 35./48. ) + 1.3125 )
             - 35./96. ) - 0.7109375 ) - 7.0/768.0;
    }
    else if ( x < 2.5 )
    {
      x = x * ( x * ( x * ( x * ( x * (-1./20.) + 7./12. )
             - 2.625 ) + 133./24. ) - 5.140625 ) + 1267./960.;
    }
    else if ( x < 3.5 )
    {
      x *= 2.;
      x -= 7.;
      scalar_t x2 = x*x;
      x = (x2 * x2 * x ) / 3840.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad6(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < .5 )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 * ( 7. / 12. ) - ( x2 * x2 ) / 6.- 77./96. );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( x * ( x * 0.125 - 35./48. ) + 1.3125 )
             - 35./96. ) - 0.7109375 ) - 7.0/768.0;
    }
    else if ( x < 2.5 )
    {
      x = x * ( x * ( x * ( x * ( x * (-1./20.) + 7./12. )
             - 2.625 ) + 133./24. ) - 5.140625 ) + 1267./960.;
    }
    else
    {
      x *= 2.;
      x -= 7.;
      scalar_t x2 = x*x;
      x = (x2 * x2 * x ) / 3840.;
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds6(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-2.5));
    upp = low + 6;
  }

  // --- order 7 -------------------------------------------------------

  template <typename scalar_t>
  static inline __device__ scalar_t weight7(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( f * ( x * ( 1. / 144. ) - 1. / 36. ) + 1. / 9. ) -
             1. / 3. ) + 151. / 315.0;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 0.05 - x * ( 1. / 240. ) ) -
             7. / 30. ) + 0.5 ) - 7. / 18. ) - 0.1 ) - 7. / 90. ) +
             103. / 210.0;
    }
    else if ( x < 3. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( x * ( 1. / 720. ) -
             1. / 36. ) + 7. / 30. ) - 19. / 18. ) + 49. / 18. ) -
             23. / 6. ) + 217. / 90. ) - 139. / 630.0;
    }
    else if ( x < 4. )
    {
      scalar_t f = 4. - x;
      x = f * f * f;
      return ( x * x * f ) / 5040.;
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastweight7(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( f * ( x * ( 1. / 144. ) - 1. / 36. ) + 1. / 9. )
             - 1. / 3. ) + 151. / 315.0;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 0.05 - x * ( 1. / 240. ) )
             - 7. / 30. ) + 0.5 ) - 7. / 18. ) - 0.1 ) - 7. / 90. )
             + 103. / 210.0;
    }
    else if ( x < 3. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( x * ( 1. / 720. )
             - 1. / 36. ) + 7. / 30. ) - 19. / 18. ) + 49. / 18. )
             - 23. / 6. ) + 217. / 90. ) - 139. / 630.0;
    }
    else
    {
      scalar_t f = 4. - x;
      x = f * f * f;
      return ( x * x * f ) / 5040.;
    }
  }

  template <typename scalar_t>
  static inline __device__ scalar_t grad7(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 *( x2 * ( x * ( 7. / 144. )
             - 1. / 6. ) + 4. / 9. ) - 2. / 3. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( x * ( x * ( -7. / 240. ) + 3. / 10. )
             - 7. / 6. ) + 2. ) - 7. / 6. ) - 1. / 5. ) - 7. / 90.;
    }
    else if ( x < 3. )
    {
      x = x * ( x * (x * ( x * ( x * ( x * ( 7. / 720. ) - 1. / 6. )
             + 7. / 6. ) - 38. / 9. ) + 49. / 6. ) - 23. / 3. ) + 217. / 90.;
    }
    else if ( x < 4. )
    {
      x -= 4;
      x *= x*x;
      x *= x;
      x = - x / 720.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline __device__ scalar_t fastgrad7(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 *( x2 * ( x * ( 7. / 144. )
             - 1. / 6. ) + 4. / 9. ) - 2. / 3. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( x * ( x * ( -7. / 240. ) + 3. / 10. )
             - 7. /6. ) + 2. ) - 7. / 6. ) - 1. / 5. ) - 7. / 90.;
    }
    else if ( x < 3. )
    {
      x = x * ( x * (x * ( x * ( x * ( x * ( 7. / 720. ) - 1. / 6. )
             + 7. / 6. ) - 38. / 9. ) + 49. / 6. ) - 23. / 3. ) + 217. / 90.;
    }
    else
    {
      x -= 4;
      x *= x*x;
      x *= x;
      x = - x / 720.;
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t, typename offset_t>
  static inline __device__ void bounds7(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-3.));
    upp = low + 7;
  }


} // namespace _spline

template <type I> struct utils {};

#define INTERPOL_UTILS(NAME, ORDER) \
template <> struct utils<type::##NAME> { \
    template <typename scalar_t> \
    static inline __device__ scalar_t \
    weight(scalar_t x) { return _spline::weight##ORDER(x); } \
    template <typename scalar_t> \
    static inline __device__ scalar_t \
    fastweight(scalar_t x) { return _spline::fastweight##ORDER(x); } \
    template <typename scalar_t> \
    static inline __device__ scalar_t \
    grad(scalar_t x) { return _spline::grad##ORDER(x); } \
    template <typename scalar_t> \
    static inline __device__ scalar_t \
    fastgrad(scalar_t x) { return _spline::fastgrad##ORDER(x); } \
    template <typename scalar_t> \
    static inline __device__ scalar_t \
    hess(scalar_t x) { return _spline::hess##ORDER(x); } \
    template <typename scalar_t> \
    static inline __device__ scalar_t \
    fasthess(scalar_t x) { return _spline::fasthess##ORDER(x); } \
    template <typename scalar_t, typename offset_t> \
    static inline __device__ void \
    bounds(scalar_t x, offset_t & low, offset_t & upp) { return _spline::bounds##ORDER(x, low, upp); } \
};

INTERPOL_UTILS(Nearest, 0)
INTERPOL_UTILS(Linear, 1)
INTERPOL_UTILS(Quadratic, 2)
INTERPOL_UTILS(Cubic, 3)
INTERPOL_UTILS(FourthOrder, 4)
INTERPOL_UTILS(FifthOrder, 5)
INTERPOL_UTILS(SixthOrder, 6)
INTERPOL_UTILS(SeventhOrder, 7)

template <typename scalar_t>
static inline __device__ scalar_t
weight(type spline_type, scalar_t x) {
  switch (spline_type) {
    case type::Nearest:      return _spline::weight0(x);
    case type::Linear:       return _spline::weight1(x);
    case type::Quadratic:    return _spline::weight2(x);
    case type::Cubic:        return _spline::weight3(x);
    case type::FourthOrder:  return _spline::weight4(x);
    case type::FifthOrder:   return _spline::weight5(x);
    case type::SixthOrder:   return _spline::weight6(x);
    case type::SeventhOrder: return _spline::weight7(x);
    default:                 return _spline::weight1(x);
  }
}

template <typename scalar_t>
static inline __device__ scalar_t
fastweight(type spline_type, scalar_t x) {
  switch (spline_type) {
    case type::Nearest:      return _spline::fastweight0(x);
    case type::Linear:       return _spline::fastweight1(x);
    case type::Quadratic:    return _spline::fastweight2(x);
    case type::Cubic:        return _spline::fastweight3(x);
    case type::FourthOrder:  return _spline::fastweight4(x);
    case type::FifthOrder:   return _spline::fastweight5(x);
    case type::SixthOrder:   return _spline::fastweight6(x);
    case type::SeventhOrder: return _spline::fastweight7(x);
    default:                 return _spline::fastweight1(x);
  }
}

template <typename scalar_t>
static inline __device__ scalar_t
grad(type spline_type, scalar_t x) {
  switch (spline_type) {
    case type::Nearest:      return _spline::grad0(x);
    case type::Linear:       return _spline::grad1(x);
    case type::Quadratic:    return _spline::grad2(x);
    case type::Cubic:        return _spline::grad3(x);
    case type::FourthOrder:  return _spline::grad4(x);
    case type::FifthOrder:   return _spline::grad5(x);
    case type::SixthOrder:   return _spline::grad6(x);
    case type::SeventhOrder: return _spline::grad7(x);
    default:                 return _spline::grad1(x);
  }
}

template <typename scalar_t>
static inline __device__ scalar_t
fastgrad(type spline_type, scalar_t x) {
  switch (spline_type) {
    case type::Nearest:      return _spline::fastgrad0(x);
    case type::Linear:       return _spline::fastgrad1(x);
    case type::Quadratic:    return _spline::fastgrad2(x);
    case type::Cubic:        return _spline::fastgrad3(x);
    case type::FourthOrder:  return _spline::fastgrad4(x);
    case type::FifthOrder:   return _spline::fastgrad5(x);
    case type::SixthOrder:   return _spline::fastgrad6(x);
    case type::SeventhOrder: return _spline::fastgrad7(x);
    default:                 return _spline::fastgrad1(x);
  }
}

template <typename scalar_t>
static inline __device__ scalar_t
hess(type spline_type, scalar_t x) {
  switch (spline_type) {
    case type::Nearest:      return _spline::hess0(x);
    case type::Linear:       return _spline::hess1(x);
    case type::Quadratic:    return _spline::hess2(x);
    case type::Cubic:        return _spline::hess3(x);
    case type::FourthOrder:  return _spline::hess4(x);
    case type::FifthOrder:   return _spline::hess0(x); // notimplemented
    case type::SixthOrder:   return _spline::hess0(x); // notimplemented
    case type::SeventhOrder: return _spline::hess0(x); // notimplemented
    default:                 return _spline::hess1(x);
  }
}

template <typename scalar_t>
static inline __device__ scalar_t
fasthess(type spline_type, scalar_t x) {
  switch (spline_type) {
    case type::Nearest:      return _spline::fasthess0(x);
    case type::Linear:       return _spline::fasthess1(x);
    case type::Quadratic:    return _spline::fasthess2(x);
    case type::Cubic:        return _spline::fasthess3(x);
    case type::FourthOrder:  return _spline::fasthess4(x);
    case type::FifthOrder:   return _spline::fasthess0(x); // notimplemented
    case type::SixthOrder:   return _spline::fasthess0(x); // notimplemented
    case type::SeventhOrder: return _spline::fasthess0(x); // notimplemented
    default:                 return _spline::fasthess1(x);
  }
}

template <typename scalar_t, typename offset_t>
static inline __device__ void
bounds(type spline_type, scalar_t x, offset_t & low, offset_t & upp)
 {
  switch (spline_type) {
    case type::Nearest:      return _spline::bounds0(x, low, upp);
    case type::Linear:       return _spline::bounds1(x, low, upp);
    case type::Quadratic:    return _spline::bounds2(x, low, upp);
    case type::Cubic:        return _spline::bounds3(x, low, upp);
    case type::FourthOrder:  return _spline::bounds4(x, low, upp);
    case type::FifthOrder:   return _spline::bounds5(x, low, upp);
    case type::SixthOrder:   return _spline::bounds6(x, low, upp);
    case type::SeventhOrder: return _spline::bounds7(x, low, upp);
    default:                 return _spline::bounds1(x, low, upp);
  }
}


} // namespace spline
} // namespace jf

#endif // JF_SPLINE
