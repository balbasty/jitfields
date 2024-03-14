#ifndef JF_VECTOR_TRAITS_H
#define JF_VECTOR_TRAITS_H

namespace jf {
namespace internal {

// This trait is used to replace `D ` with `DerivedType` if it is defined
// (i.e. not "void"), otherwise with itself.
template <typename BaseType, typename DerivedType>
struct guess_type
{
    using value = DerivedType;
};

template <typename BaseType>
struct guess_type<BaseType, void>
{
    using value = BaseType;
};

// This trait is used ot know if we should inherit a type from the parent
// or from the derived implementation
template <typename DerivedType, typename DerivedUsing, typename BaseUsing>
struct inherit_type
{
    using value = DerivedType;
};

template <typename DerivedUsing, typename BaseUsing>
struct inherit_type<void, DerivedUsing, BaseUsing>
{
    using value = BaseUsing;
};

template <typename DerivedType, typename T, T DerivedUsing, T BaseUsing>
struct inherit_expr
{
    static constexpr T value = DerivedUsing;
};

template <typename T, T DerivedUsing, T BaseUsing>
struct inherit_expr<void, T, DerivedUsing, BaseUsing>
{
    static constexpr T value = BaseUsing;
};

// This trait is used to guess iterator types
template <typename BaseType, typename DerivedType>
struct guess_iterator {
    using forward = typename DerivedType::iterator;
    using const_forward = typename DerivedType::const_iterator;
    using reverse = typename DerivedType::reverse_iterator;
    using const_reverse = typename DerivedType::const_reverse_iterator;
};
template <typename BaseType>
struct guess_iterator<BaseType, void> {
    using forward = WeakRef<typename BaseType::value_type, BaseType::static_stride>;
    using const_forward = WeakRef<const typename BaseType::value_type, BaseType::static_stride>;
    using reverse = WeakRef<typename BaseType::value_type, -BaseType::static_stride>;
    using const_reverse = WeakRef<const typename BaseType::value_type, -BaseType::static_stride>;
};

} // internal
} // jf

#endif // JF_VECTOR_TRAITS_H
