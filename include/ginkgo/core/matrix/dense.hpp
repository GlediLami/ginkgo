// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType>
class Diagonal;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Fbcsr;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;

template <typename ValueType, typename IndexType>
class SparsityCsr;


/**
 * Dense is a matrix format which explicitly stores all values of the
 * matrix.
 *
 * The values are stored in row-major format (values belonging to the same row
 * appear consecutive in the memory). Optionally, rows can be padded for better
 * memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup dense
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Dense : public EnableLinOp<Dense<ValueType>>,
              public ConvertibleTo<Dense<next_precision<ValueType>>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
              public ConvertibleTo<Dense<next_precision<ValueType, 2>>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
              public ConvertibleTo<Dense<next_precision<ValueType, 3>>>,
#endif
              public ConvertibleTo<Coo<ValueType, int32>>,
              public ConvertibleTo<Coo<ValueType, int64>>,
              public ConvertibleTo<Csr<ValueType, int32>>,
              public ConvertibleTo<Csr<ValueType, int64>>,
              public ConvertibleTo<Ell<ValueType, int32>>,
              public ConvertibleTo<Ell<ValueType, int64>>,
              public ConvertibleTo<Fbcsr<ValueType, int32>>,
              public ConvertibleTo<Fbcsr<ValueType, int64>>,
              public ConvertibleTo<Hybrid<ValueType, int32>>,
              public ConvertibleTo<Hybrid<ValueType, int64>>,
              public ConvertibleTo<Sellp<ValueType, int32>>,
              public ConvertibleTo<Sellp<ValueType, int64>>,
              public ConvertibleTo<SparsityCsr<ValueType, int32>>,
              public ConvertibleTo<SparsityCsr<ValueType, int64>>,
              public ReadableFromMatrixData<ValueType, int32>,
              public ReadableFromMatrixData<ValueType, int64>,
              public WritableToMatrixData<ValueType, int32>,
              public WritableToMatrixData<ValueType, int64> {
    friend class EnablePolymorphicObject<Dense, LinOp>;
    friend class Coo<ValueType, int32>;
    friend class Coo<ValueType, int64>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Diagonal<ValueType>;
    friend class Ell<ValueType, int32>;
    friend class Ell<ValueType, int64>;
    friend class Fbcsr<ValueType, int32>;
    friend class Fbcsr<ValueType, int64>;
    friend class Hybrid<ValueType, int32>;
    friend class Hybrid<ValueType, int64>;
    friend class Sellp<ValueType, int32>;
    friend class Sellp<ValueType, int64>;
    friend class SparsityCsr<ValueType, int32>;
    friend class SparsityCsr<ValueType, int64>;
    friend class Dense<to_complex<ValueType>>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableLinOp<Dense>::convert_to;
    using EnableLinOp<Dense>::move_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType>>>::move_to;
    using ConvertibleTo<Coo<ValueType, int32>>::convert_to;
    using ConvertibleTo<Coo<ValueType, int32>>::move_to;
    using ConvertibleTo<Coo<ValueType, int64>>::convert_to;
    using ConvertibleTo<Coo<ValueType, int64>>::move_to;
    using ConvertibleTo<Csr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int32>>::move_to;
    using ConvertibleTo<Csr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int64>>::move_to;
    using ConvertibleTo<Ell<ValueType, int32>>::convert_to;
    using ConvertibleTo<Ell<ValueType, int32>>::move_to;
    using ConvertibleTo<Ell<ValueType, int64>>::convert_to;
    using ConvertibleTo<Ell<ValueType, int64>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, int32>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, int64>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, int32>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, int32>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, int64>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, int64>>::move_to;
    using ConvertibleTo<Sellp<ValueType, int32>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, int32>>::move_to;
    using ConvertibleTo<Sellp<ValueType, int64>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, int64>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, int32>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, int32>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, int64>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, int64>>::move_to;
    using ReadableFromMatrixData<ValueType, int32>::read;
    using ReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int64;
    using transposed_type = Dense<value_type>;
    using mat_data = matrix_data<value_type, int64>;
    using mat_data32 = matrix_data<value_type, int32>;
    using device_mat_data = device_matrix_data<value_type, int64>;
    using device_mat_data32 = device_matrix_data<value_type, int32>;
    using absolute_type = remove_complex<Dense>;
    using real_type = absolute_type;
    using complex_type = to_complex<Dense>;
    using device_view = view::dense<value_type>;
    using const_device_view = view::dense<const value_type>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    friend class MultiVector<previous_precision<ValueType>>;

    void convert_to(
        MultiVector<next_precision<ValueType>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType>>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class MultiVector<previous_precision<ValueType, 2>>;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 2>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 2>>>::move_to;

    void convert_to(
        MultiVector<next_precision<ValueType, 2>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType, 2>>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class MultiVector<previous_precision<ValueType, 3>>;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 3>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 3>>>::move_to;

    void convert_to(
        MultiVector<next_precision<ValueType, 3>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType, 3>>* result) override;
#endif

    void convert_to(Coo<ValueType, int32>* result) const override;

    void move_to(Coo<ValueType, int32>* result) override;

    void convert_to(Coo<ValueType, int64>* result) const override;

    void move_to(Coo<ValueType, int64>* result) override;

    void convert_to(Csr<ValueType, int32>* result) const override;

    void move_to(Csr<ValueType, int32>* result) override;

    void convert_to(Csr<ValueType, int64>* result) const override;

    void move_to(Csr<ValueType, int64>* result) override;

    void convert_to(Ell<ValueType, int32>* result) const override;

    void move_to(Ell<ValueType, int32>* result) override;

    void convert_to(Ell<ValueType, int64>* result) const override;

    void move_to(Ell<ValueType, int64>* result) override;

    void convert_to(Fbcsr<ValueType, int32>* result) const override;

    void move_to(Fbcsr<ValueType, int32>* result) override;

    void convert_to(Fbcsr<ValueType, int64>* result) const override;

    void move_to(Fbcsr<ValueType, int64>* result) override;

    void convert_to(Hybrid<ValueType, int32>* result) const override;

    void move_to(Hybrid<ValueType, int32>* result) override;

    void convert_to(Hybrid<ValueType, int64>* result) const override;

    void move_to(Hybrid<ValueType, int64>* result) override;

    void convert_to(Sellp<ValueType, int32>* result) const override;

    void move_to(Sellp<ValueType, int32>* result) override;

    void convert_to(Sellp<ValueType, int64>* result) const override;

    void move_to(Sellp<ValueType, int64>* result) override;

    void convert_to(SparsityCsr<ValueType, int32>* result) const override;

    void move_to(SparsityCsr<ValueType, int32>* result) override;

    void convert_to(SparsityCsr<ValueType, int64>* result) const override;

    void move_to(SparsityCsr<ValueType, int64>* result) override;

    void read(const mat_data& data) override;

    void read(const mat_data32& data) override;

    void read(const device_mat_data& data) override;

    void read(const device_mat_data32& data) override;

    void read(device_mat_data&& data) override;

    void read(device_mat_data32&& data) override;

    void write(mat_data& data) const override;

    void write(mat_data32& data) const override;

protected:
    void apply_impl(const IMultiVector* b, IMultiVector* x) const override;
    void apply_impl(const IMultiVector* alpha, const IMultiVector* b,
                    const IMultiVector* beta, IMultiVector* x) const override;
};

}  // namespace matrix
}  // namespace gko
