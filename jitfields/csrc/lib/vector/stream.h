#ifndef JF_VECTOR_STREAM_H
#define JF_VECTOR_STREAM_H
#include <iostream>
#include "abstract_ptr.h"
#include "abstract_sized.h"
#include "abstract_vector.h"
#include "weak_ref.h"
#include "weak_sized.h"
#include "weak_vector.h"
#include "concrete_vector.h"

template <typename T>
struct is_void { static constexpr bool value = false; };

template <>
struct is_void<void> { static constexpr bool value = true; };


template <typename T, long S, typename D>
std::ostream& AbstractPointer_print(
    std::ostream& stream, const jf::AbstractPointer<T,S,D> & ptr)
{
    stream << "AbstractPointer" << std::endl;
    stream << "- data:   " << ptr.data() << std::endl;
    stream << "- stride: " << ptr.stride();
    if (S == 0) stream << " (dynamic)";
    stream << std::endl;
    stream << "- final:  " << (is_void<D>::value ? "yes" : "no") << std::endl;
    return stream;
}

template <typename T, long S, typename D>
std::ostream& operator <<(
    std::ostream& stream, const jf::AbstractPointer<T,S,D> & ptr)
{
    return AbstractPointer_print(stream, ptr);
}


template <typename T, unsigned long N, long S, typename D>
std::ostream& AbstractSizedPointer_print(
    std::ostream& stream, const jf::AbstractSizedPointer<T,N,S,D> & ptr)
{
    stream << "AbstractSizedPointer" << std::endl;
    stream << "- data:   " << ptr.data() << std::endl;
    stream << "- size:   " << ptr.size();
    if (N == 0) stream << " (dynamic)";
    stream << std::endl;
    stream << "- stride: " << ptr.stride();
    if (S == 0) stream << " (dynamic)";
    stream << std::endl;
    stream << "- final:  " << (is_void<D>::value ? "yes" : "no") << std::endl;
    return stream;
}

template <typename T, unsigned long N, long S, typename D>
std::ostream& operator <<(std::ostream& stream,
                          const jf::AbstractSizedPointer<T,N,S,D> & ptr)
{
    return AbstractSizedPointer_print(stream, ptr);
}


template <typename T, unsigned long N, long S, typename D>
std::ostream& AbstractVector_print(
    std::ostream& stream, const jf::AbstractVector<T,N,S,D> & ptr)
{
    return AbstractSizedPointer_print(stream << "AbstractVector [cast to] ", ptr.ref());
}

template <typename T, unsigned long N, long S, typename D>
std::ostream& operator <<(std::ostream& stream,
                          const jf::AbstractVector<T,N,S,D> & ptr)
{
    return AbstractVector_print(stream, ptr);
}


template <typename T, long S, typename D>
std::ostream& WeakRef_print(
    std::ostream& stream, const jf::WeakRef<T,S,D> & ptr)
{
    return AbstractPointer_print(stream << "WeakRef: ", ptr);
}

template <typename T, long S, typename D>
std::ostream& operator <<(std::ostream& stream,
                          const jf::WeakRef<T,S,D> & ptr)
{
    return WeakRef_print(stream, ptr);
}


template <typename T, unsigned long N, long S, typename D>
std::ostream& WeakSizedRef_print(
    std::ostream& stream, const jf::WeakSizedRef<T,N,S,D> & ptr)
{
    return AbstractSizedPointer_print(stream << "WeakSizedRef: ", ptr);
}

template <typename T, unsigned long N, long S, typename D>
std::ostream& operator <<(
    std::ostream& stream, const jf::WeakSizedRef<T,N,S,D> & ptr)
{
    return WeakSizedRef_print(stream, ptr);
}


template <typename T, unsigned long N, long S, typename D>
std::ostream& WeakVector_print(
    std::ostream& stream, const jf::WeakVector<T,N,S,D> & ptr)
{
    return AbstractVector_print(stream << "WeakVector: ", ptr);
}

template <typename T, unsigned long N, long S, typename D>
std::ostream& operator <<(
    std::ostream& stream, const jf::WeakVector<T,N,S,D> & ptr)
{
    return WeakVector_print(stream, ptr);
}


template <typename T, unsigned long N, typename D>
std::ostream& Vector_print(
    std::ostream& stream, const jf::Vector<T,N,D> & ptr)
{
    return AbstractVector_print(stream << "Vector: ", ptr);
}

template <typename T, unsigned long N, typename D>
std::ostream& operator <<(
    std::ostream& stream, const jf::Vector<T,N,D> & ptr)
{
    return Vector_print(stream, ptr);
}

#endif // JF_VECTOR_STREAM_H
