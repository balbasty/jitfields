#ifndef JF_VECTOR_VECTOR_H
#define JF_VECTOR_VECTOR_H

#include "weak_ref.h"
#include "weak_sized.h"
#include "weak_vector.h"
#include "concrete_vector.h"

/* TYPE HIERARCHY
 *
 *    AbstractPointer
 *    |- WeakRef
 *    |- AbstractSizedPointer
 *       |- WeakSizedRef
 *
 *    AbstractVector
 *    |- WeakVector
 *    |- Vector
 *
 * The main difference between "SizedRefs" and "Vectors" is that "SizedRefs"
 * act like native pointers, in the sense that dereference do not change
 * the constness of the object itself.
 * Furthermore, "Refs" understand pointer arithmetic.
 * I.e., a `const SizedRef<int> ptr` is equivalent to a `int * const ptr`,
 * and a `SizedRef<const int> ptr` is  is equivalent to a `cont int * ptr`.
 *
 * On the other hand:
 * - a `const WeakVector<int> ptr` acts more like a `const int * const ptr`
 * - a `WeakVector<const int> ptr` acts more like a `int * const ptr`
 * - a `WeakVector<int> ptr` acts more like a `int * ptr`
 * However, "Vectors" do not understand pointer arithmetic. Iterators
 * (or conversion to "Refs" using the ref() method) should be used for
 * pointer-like behavior.
 */

/* WEAK REFERENCE
template <ValueType, Stride = 1>
class WeakRef: AbstractPointer
{
    // info
    data(), stride()

    // constructor
    WeakRef(data = nullptr, stride = Stride)

    // conversion
    operator ValueType* ()

    // accessors
    operator[](i)

    // iterators
    begin(), cbegin(), end(), cend()

    // increment
    operator++(), operator--(), operator+=(i), operator-=(i)
};
*/

/* WEAK SIZED REFERENCE
template <ValueType, Size = 0, Stride = 1>
class WeakSizedRef: AbstractSizedPointer
{
public:
    // info
    data(), size(), stride()

    // constructor
    WeakSizedRef(size = Size, data = nullptr, stride = Stride)
    WeakSizedRef(data, stride = Stride)

    // conversion
    operator ValueType* ()
    operator WeakRef<ValueType> ()
    operator WeakVector<ValueType> ()

    // accessors
    operator[](i)

    // iterators
    begin(), cbegin(), rbegin(), crbegin(), end(), cend(), rend(), crend()

    // increment
    operator++(), operator--(), operator+=(i), operator-=(i)
    operator+(i), operator-(i)
};
*/

/* WEAK VECTOR
template <ValueType, Size = 0, Stride = 1>
class WeakVector: AbstractVector
{
public:
    // info
    data(), size(), stride()

    // constructor
    WeakVector(size = Size, data = nullptr, stride = Stride)
    WeakVector(data, stride = Stride)

    // conversion
    operator ValueType* ()
    operator WeakSizedRef<ValueType> ()

    // accessors
    operator[](i)

    // iterators
    begin(), cbegin(), rbegin(), crbegin(), end(), cend(), rend(), crend()
};
*/

/* VECTOR
template <ValueType, Size = 0, Stride = 1>
class Vector: AbstractVector
{
public:
    // info
    data(), size(), stride()

    // constructor
    Vector(size = Size, data = nullptr, stride = Stride)
    Vector(data, stride = Stride)

    // conversion
    operator ValueType* ()
    operator WeakSizedRef<ValueType> ()
    operator WeakVector<ValueType> ()

    // accessors
    operator[](i)

    // iterators
    begin(), cbegin(), rbegin(), crbegin(), end(), cend(), rend(), crend()
};
*/

#endif // JF_VECTOR_VECTOR_H
