#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================

        auto shape_A = inputs[0]->getDims();
        auto shape_B = inputs[1]->getDims();
        auto rank_A = shape_A.size();
        auto rank_B = shape_B.size();

        if (transA && rank_A >= 2) {
            std::swap(shape_A[rank_A - 1], shape_A[rank_A - 2]);
        }
        if (transB && rank_B >= 2) {
            std::swap(shape_B[rank_B - 1], shape_B[rank_B - 2]);
        }

        shape_A[rank_A - 1] = 1;
        shape_B[rank_B - 2] = 1;

        auto shape_output = infer_broadcast(shape_A, shape_B);

        return { {shape_output} };
    }

} // namespace infini