// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <../../include/ginkgo/core/base/multivector.hpp>

#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {


IMultiVector::IMultiVector(std::shared_ptr<const Executor> exec,
                           const dim<2>& size, precision p)
    : EnableAbstractPolymorphicObject(std::move(exec)),
      size_(size),
      precision_(p)
{}


void IMultiVector::set_size(const dim<2>& value) noexcept { size_ = value; }


std::unique_ptr<IMultiVector> IMultiVector::create_with_config_of(
    ptr_param<const IMultiVector> other)
{
    return other->create_generic_with_same_config_impl();
}


std::unique_ptr<IMultiVector> IMultiVector::create_with_type_of(
    ptr_param<const IMultiVector> other, std::shared_ptr<const Executor> exec)
{
    return other->create_generic_with_type_of_impl(std::move(exec), {}, {}, 0);
}


std::unique_ptr<IMultiVector> IMultiVector::create_with_type_of(
    ptr_param<const IMultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size)
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
    return other->create_generic_with_type_of_impl(std::move(exec), global_size,
                                                   local_size, global_size[1]);
}


std::unique_ptr<IMultiVector> IMultiVector::create_with_type_of(
    ptr_param<const IMultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size, size_type stride)
{
    return other->create_generic_with_type_of_impl(std::move(exec), global_size,
                                                   local_size, stride);
}


std::unique_ptr<IMultiVector> IMultiVector::compute_absolute() const
{
    return this->compute_absolute_generic_impl();
}


void IMultiVector::compute_absolute(ptr_param<IMultiVector> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    this->compute_absolute_generic_impl(output.get());
}


void IMultiVector::compute_absolute_inplace()
{
    this->compute_absolute_inplace_impl();
}


std::unique_ptr<IMultiVector> IMultiVector::make_complex() const
{
    return this->make_complex_generic_impl();
}


void IMultiVector::make_complex(ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    this->make_complex_generic_impl(result.get());
}


std::unique_ptr<IMultiVector> IMultiVector::get_real() const
{
    return this->get_real_generic_impl();
}


void IMultiVector::get_real(ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    this->get_real_generic_impl(result.get());
}


std::unique_ptr<IMultiVector> IMultiVector::get_imag() const
{
    return this->get_imag_generic_impl();
}


void IMultiVector::get_imag(ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    this->get_imag_generic_impl(result.get());
}


void IMultiVector::fill(any_scalar value) { this->fill_impl(value); }


#define GKO_ASSERT_IS_DENSE(alpha)                                           \
    {                                                                        \
        bool is_dense = std::visit(                                          \
            [alpha](auto p) {                                                \
                using value_type = std::decay_t<decltype(p)>;                \
                return dynamic_cast<const matrix::MultiVector<value_type>*>( \
                           alpha.get()) != nullptr;                          \
            },                                                               \
            precision_to_variant(alpha->get_precision()));                   \
        if (!is_dense) {                                                     \
            GKO_NOT_SUPPORTED(alpha);                                        \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


void IMultiVector::scale(ptr_param<const IMultiVector> alpha)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    this->scale_impl(alpha.get());
}


void IMultiVector::inv_scale(ptr_param<const IMultiVector> alpha)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    this->inv_scale_impl(alpha.get());
}


void IMultiVector::add_scaled(ptr_param<const IMultiVector> alpha,
                              ptr_param<const IMultiVector> b)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    this->add_scaled_impl(alpha.get(), b.get());
}


void IMultiVector::sub_scaled(ptr_param<const IMultiVector> alpha,
                              ptr_param<const IMultiVector> b)
{
    GKO_ASSERT_IS_DENSE(alpha);
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    this->sub_scaled_impl(alpha.get(), b.get());
}


void IMultiVector::compute_dot(ptr_param<const IMultiVector> b,
                               ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_dot_impl(b.get(), result.get());
}


void IMultiVector::compute_dot(ptr_param<const IMultiVector> b,
                               ptr_param<IMultiVector> result,
                               array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_dot_impl(b.get(), result.get(), tmp);
}


void IMultiVector::compute_conj_dot(ptr_param<const IMultiVector> b,
                                    ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_conj_dot_impl(b.get(), result.get());
}


void IMultiVector::compute_conj_dot(ptr_param<const IMultiVector> b,
                                    ptr_param<IMultiVector> result,
                                    array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_conj_dot_impl(b.get(), result.get(), tmp);
}


void IMultiVector::compute_norm2(ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_norm2_impl(result.get());
}


void IMultiVector::compute_norm2(ptr_param<IMultiVector> result,
                                 array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_norm2_impl(result.get(), tmp);
}


void IMultiVector::compute_squared_norm2(ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_squared_norm2_impl(result.get());
}


void IMultiVector::compute_squared_norm2(ptr_param<IMultiVector> result,
                                         array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_squared_norm2_impl(result.get(), tmp);
}


void IMultiVector::compute_norm1(ptr_param<IMultiVector> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_norm1_impl(result.get());
}


void IMultiVector::compute_norm1(ptr_param<IMultiVector> result,
                                 array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    this->compute_norm1_impl(result.get(), tmp);
}


std::unique_ptr<const IMultiVector> IMultiVector::create_real_view() const
{
    return this->create_real_view_generic_impl();
}


std::unique_ptr<IMultiVector> IMultiVector::create_real_view()
{
    return this->create_real_view_generic_impl();
}


std::unique_ptr<IMultiVector> IMultiVector::create_subview(local_span rows,
                                                           local_span columns)
{
    return this->create_subview_generic_impl(rows, columns);
}


std::unique_ptr<const IMultiVector> IMultiVector::create_subview(
    local_span rows, local_span columns) const
{
    return this->create_subview_generic_impl(rows, columns);
}


std::unique_ptr<const IMultiVector> IMultiVector::create_subview(
    local_span rows, local_span columns, dim<2> global_size) const
{
    return this->create_subview_generic_impl(rows, columns, global_size);
}


std::unique_ptr<IMultiVector> IMultiVector::create_subview(local_span rows,
                                                           local_span columns,
                                                           dim<2> global_size)
{
    return this->create_subview_generic_impl(rows, columns, global_size);
}


gko::detail::temporary_conversion<IMultiVector> IMultiVector::as_precision(
    precision p)
{
    return this->as_precision_impl(p);
}


detail::temporary_conversion<IMultiVector> IMultiVector::as_precision(
    ptr_param<const IMultiVector> p)
{
    return this->as_precision_impl(p->get_precision());
}


detail::temporary_conversion<IMultiVector> IMultiVector::as_precision(
    ptr_param<const LinOp> p)
{
    return this->as_precision_impl(p->get_precision());
}


gko::detail::temporary_conversion<const IMultiVector>
IMultiVector::as_precision(precision p) const
{
    return this->as_precision_impl(p);
}


detail::temporary_conversion<const IMultiVector> IMultiVector::as_precision(
    ptr_param<const IMultiVector> p) const
{
    return this->as_precision_impl(p->get_precision());
}


detail::temporary_conversion<const IMultiVector> IMultiVector::as_precision(
    ptr_param<const LinOp> p) const
{
    return this->as_precision_impl(p->get_precision());
}


precision IMultiVector::get_precision() const noexcept { return precision_; }


dim<2> IMultiVector::get_size() const noexcept { return size_; }


IMultiVector::IMultiVector(const IMultiVector& other)
    : EnableAbstractPolymorphicObject(other),
      size_(other.size_),
      precision_(other.precision_)
{}


IMultiVector::IMultiVector(IMultiVector&& other)
    : EnableAbstractPolymorphicObject(std::move(other)),
      size_(std::exchange(other.size_, {})),
      precision_(other.precision_)
{}


IMultiVector& IMultiVector::operator=(const IMultiVector& other)
{
    if (this != &other) {
        EnableAbstractPolymorphicObject::operator=(other);
        size_ = other.size_;
    }
    return *this;
}


IMultiVector& IMultiVector::operator=(IMultiVector&& other)
{
    if (this != &other) {
        EnableAbstractPolymorphicObject::operator=(std::move(other));
        size_ = std::exchange(other.size_, {});
    }
    return *this;
}


template <typename ValueType>
IMultiVector::device_view<ValueType> IMultiVector::get_local_device_view()
{
    if (this->get_precision() != type_to_precision<ValueType>) {
        GKO_INVALID_STATE("Multivector doesn't have the requested precision");
    }
    using return_type = device_view<ValueType>;
    auto variant = this->get_local_device_view_generic_impl();
    return std::move(std::get<return_type>(variant));
}

#define GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW(ValueType) \
    IMultiVector::device_view<ValueType> IMultiVector::get_local_device_view()
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW);


template <typename ValueType>
IMultiVector::device_view<const ValueType>
IMultiVector::get_const_local_device_view() const
{
    if (this->get_precision() != type_to_precision<ValueType>) {
        GKO_INVALID_STATE("Multivector doesn't have the requested precision");
    }
    using return_type = device_view<const ValueType>;
    auto variant = this->get_const_local_device_view_generic_impl();
    return std::move(std::get<return_type>(variant));
}

#define GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW_CONST(ValueType) \
    IMultiVector::device_view<const ValueType>                     \
    IMultiVector::get_const_local_device_view() const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CREATE_LOCAL_VIEW_CONST);


}  // namespace gko
