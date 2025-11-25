#include <gtest/gtest.h>
#include "fock_ell_operator.h"
#include <vector>
#include <cmath>

/**
 * FockELLOperator 单元测试
 */
class FockELLOperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 8;
        bandwidth = 3;
        ell_op = new FockELLOperator(dim, bandwidth);
    }

    void TearDown() override {
        delete ell_op;
    }

    int dim;
    int bandwidth;
    FockELLOperator* ell_op;
};

TEST_F(FockELLOperatorTest, Initialization) {
    EXPECT_EQ(ell_op->dim, dim);
    EXPECT_EQ(ell_op->max_bandwidth, bandwidth);
    EXPECT_FALSE(ell_op->is_empty());
}

TEST_F(FockELLOperatorTest, BuildFromDenseIdentity) {
    // 创建单位矩阵
    std::vector<cuDoubleComplex> identity(dim * dim, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < dim; ++i) {
        identity[i * dim + i] = make_cuDoubleComplex(1.0, 0.0);
    }

    ell_op->build_from_dense(identity);

    // 验证对角元素
    for (int i = 0; i < dim; ++i) {
        cuDoubleComplex elem = ell_op->get_element(i, i);
        EXPECT_NEAR(cuCreal(elem), 1.0, 1e-10);
        EXPECT_NEAR(cuCimag(elem), 0.0, 1e-10);
    }

    // 验证非对角元素为0
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i != j) {
                cuDoubleComplex elem = ell_op->get_element(i, j);
                EXPECT_NEAR(cuCreal(elem), 0.0, 1e-10);
                EXPECT_NEAR(cuCimag(elem), 0.0, 1e-10);
            }
        }
    }
}

TEST_F(FockELLOperatorTest, BuildFromDenseDiagonal) {
    // 创建对角矩阵
    std::vector<cuDoubleComplex> diagonal_matrix(dim * dim, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < dim; ++i) {
        double value = static_cast<double>(i + 1);
        diagonal_matrix[i * dim + i] = make_cuDoubleComplex(value, 0.0);
    }

    ell_op->build_from_dense(diagonal_matrix);

    // 验证对角元素
    for (int i = 0; i < dim; ++i) {
        cuDoubleComplex elem = ell_op->get_element(i, i);
        double expected = static_cast<double>(i + 1);
        EXPECT_NEAR(cuCreal(elem), expected, 1e-10);
        EXPECT_NEAR(cuCimag(elem), 0.0, 1e-10);
    }
}

TEST_F(FockELLOperatorTest, BuildFromDiagonals) {
    // 创建带状矩阵：主对角线 + 次对角线
    std::vector<std::vector<cuDoubleComplex>> diagonals = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},  // 主对角线
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},        // 上对角线
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}         // 下对角线
    };
    std::vector<int> offsets = {0, 1, -1};

    ell_op->build_from_diagonals(diagonals, offsets);

    // 验证主对角线
    for (int i = 0; i < dim; ++i) {
        cuDoubleComplex elem = ell_op->get_element(i, i);
        double expected = static_cast<double>(i + 1);
        EXPECT_NEAR(cuCreal(elem), expected, 1e-10);
    }

    // 验证上对角线
    for (int i = 0; i < dim - 1; ++i) {
        cuDoubleComplex elem = ell_op->get_element(i, i + 1);
        double expected = 0.1 * (i + 1);
        EXPECT_NEAR(cuCreal(elem), expected, 1e-10);
    }

    // 验证下对角线
    for (int i = 1; i < dim; ++i) {
        cuDoubleComplex elem = ell_op->get_element(i, i - 1);
        double expected = 0.1 * i;
        EXPECT_NEAR(cuCreal(elem), expected, 1e-10);
    }
}

TEST_F(FockELLOperatorTest, SetGetElement) {
    // 手动设置矩阵元素
    ell_op->set_element(0, 0, make_cuDoubleComplex(1.0, 0.0));
    ell_op->set_element(1, 1, make_cuDoubleComplex(2.0, 1.0));
    ell_op->set_element(0, 1, make_cuDoubleComplex(0.5, -0.5));

    // 验证设置的元素
    cuDoubleComplex elem00 = ell_op->get_element(0, 0);
    EXPECT_NEAR(cuCreal(elem00), 1.0, 1e-10);
    EXPECT_NEAR(cuCimag(elem00), 0.0, 1e-10);

    cuDoubleComplex elem11 = ell_op->get_element(1, 1);
    EXPECT_NEAR(cuCreal(elem11), 2.0, 1e-10);
    EXPECT_NEAR(cuCimag(elem11), 1.0, 1e-10);

    cuDoubleComplex elem01 = ell_op->get_element(0, 1);
    EXPECT_NEAR(cuCreal(elem01), 0.5, 1e-10);
    EXPECT_NEAR(cuCimag(elem01), -0.5, 1e-10);
}

TEST_F(FockELLOperatorTest, UploadDownload) {
    // 设置一些元素
    ell_op->set_element(0, 0, make_cuDoubleComplex(1.0, 0.0));
    ell_op->set_element(1, 2, make_cuDoubleComplex(0.5, 0.5));

    // 上传到GPU
    ell_op->upload_to_gpu();

    // 修改主机端数据
    ell_op->set_element(0, 0, make_cuDoubleComplex(2.0, 0.0));

    // 下载回主机
    ell_op->download_from_gpu();

    // 验证数据是否正确下载
    cuDoubleComplex elem00 = ell_op->get_element(0, 0);
    EXPECT_NEAR(cuCreal(elem00), 1.0, 1e-10);  // 应该是原始值
    EXPECT_NEAR(cuCimag(elem00), 0.0, 1e-10);
}

TEST_F(FockELLOperatorTest, NonZeroCount) {
    ell_op->set_element(0, 0, make_cuDoubleComplex(1.0, 0.0));
    ell_op->set_element(1, 1, make_cuDoubleComplex(1.0, 0.0));
    ell_op->set_element(2, 2, make_cuDoubleComplex(1.0, 0.0));

    int nnz = ell_op->get_nnz();
    EXPECT_EQ(nnz, 3);
}

TEST_F(FockELLOperatorTest, BoundaryConditions) {
    // 测试边界条件
    ell_op->set_element(-1, 0, make_cuDoubleComplex(1.0, 0.0));  // 无效行
    ell_op->set_element(0, -1, make_cuDoubleComplex(1.0, 0.0));  // 无效列
    ell_op->set_element(dim, 0, make_cuDoubleComplex(1.0, 0.0)); // 超出范围的行
    ell_op->set_element(0, dim, make_cuDoubleComplex(1.0, 0.0)); // 超出范围的列

    // 所有元素应该仍然是默认值
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            cuDoubleComplex elem = ell_op->get_element(i, j);
            EXPECT_NEAR(cuCreal(elem), 0.0, 1e-10);
            EXPECT_NEAR(cuCimag(elem), 0.0, 1e-10);
        }
    }
}
