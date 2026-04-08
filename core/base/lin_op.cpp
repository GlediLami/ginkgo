// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/multivector.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {


void LinOp::apply(ptr_param<const IMultiVector> b,
                  ptr_param<IMultiVector> x) const
{
    this->template log<log::Logger::linop_apply_started>(this, b.get(),
                                                         x.get());
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_apply_completed>(this, b.get(),
                                                           x.get());
}


void LinOp::apply(ptr_param<const IMultiVector> alpha,
                  ptr_param<const IMultiVector> b,
                  ptr_param<const IMultiVector> beta,
                  ptr_param<IMultiVector> x) const

{
    this->template log<log::Logger::linop_advanced_apply_started>(
        this, alpha.get(), b.get(), beta.get(), x.get());
    this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                          x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_advanced_apply_completed>(
        this, alpha.get(), b.get(), beta.get(), x.get());
}


const dim<2>& LinOp::get_size() const noexcept { return size_; }


bool LinOp::apply_uses_initial_guess() const { return false; }


bool LinOp::supports_mixed_precision() const noexcept { return false; }


precision LinOp::get_precision() const noexcept { return value_t_; }


LinOp& LinOp::operator=(LinOp&& other)
{
    if (this != &other) {
        EnableAbstractPolymorphicObject<LinOp>::operator=(std::move(other));
        this->set_size(other.get_size());
        other.set_size({});
    }
    return *this;
}


LinOp::LinOp(LinOp&& other)
    : EnableAbstractPolymorphicObject<LinOp>(std::move(other)),
      size_{std::exchange(other.size_, dim<2>{})}
{}


LinOp::LinOp(std::shared_ptr<const Executor> exec, const dim<2>& size,
             precision p)
    : EnableAbstractPolymorphicObject<LinOp>(exec), size_{size}, value_t_(p)
{}


void LinOp::set_size(const dim<2>& value) noexcept { size_ = value; }


void LinOp::apply_impl(const IMultiVector* alpha, const IMultiVector* b,
                       const IMultiVector* beta, IMultiVector* x) const
{
    auto converted_b = b->as_precision(this);
    auto converted_x = x->as_precision(this);
    std::visit(
        [&](auto p) {
            using value_type = std::decay_t<decltype(p)>;
            auto dense_alpha =
                as<matrix::MultiVector<value_type>>(alpha->as_precision(this));
            auto dense_beta =
                as<matrix::MultiVector<value_type>>(beta->as_precision(this));
            auto x_clone = converted_x->clone();
            this->apply_impl(converted_b.get(), x_clone.get());
            converted_x->scale(dense_beta.get());
            converted_x->add_scaled(dense_alpha.get(), x_clone);
        },
        precision_to_variant(this->get_precision()));
}


void LinOp::validate_application_parameters(const IMultiVector* b,
                                            const IMultiVector* x) const

{
    GKO_ASSERT_CONFORMANT(this, b);
    GKO_ASSERT_EQUAL_ROWS(this, x);
    GKO_ASSERT_EQUAL_COLS(b, x);
}


void LinOp::validate_application_parameters(const LinOp* b,
                                            const LinOp* x) const
{
    GKO_ASSERT_CONFORMANT(this, b);
    GKO_ASSERT_EQUAL_ROWS(this, x);
    GKO_ASSERT_EQUAL_COLS(b, x);
}


void LinOp::validate_application_parameters(const IMultiVector* alpha,
                                            const IMultiVector* b,
                                            const IMultiVector* beta,
                                            const IMultiVector* x) const

{
    this->validate_application_parameters(b, x);
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
}


void ScaledIdentityAddable::add_scaled_identity(ptr_param<const IMultiVector> a,
                                                ptr_param<const IMultiVector> b)
{
    GKO_ASSERT_IS_SCALAR(a);
    GKO_ASSERT_IS_SCALAR(b);
    auto ae = make_temporary_clone(as<LinOp>(this)->get_executor(), a);
    auto be = make_temporary_clone(as<LinOp>(this)->get_executor(), b);
    add_scaled_identity_impl(ae.get(), be.get());
}


}  // namespace gko
