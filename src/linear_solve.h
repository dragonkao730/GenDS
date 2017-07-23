#pragma once

#include <vector>
#include <tuple>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

// c_0 x_0 + c_1 x_1 + c_3 x_3 ... = b
struct Constrain
{
    vector<pair<int, double>> coefficients;
    double b;
    Constrain() : b(0) {}
};

// AX = B
typedef vector<Constrain> LinearSystem;

// solve linear
static VectorXd /*vector<double>*/ linearSolve(const LinearSystem& linear_system, const int& n_unknows)
{
    const int n_constrains = linear_system.size();

    // init A
    SparseMatrix<double> A(n_constrains, n_unknows);
    int nnz = 0;
    for (auto const &constrain : linear_system)
        nnz += constrain.coefficients.size();
    A.reserve(nnz);

    // fill A
    for (int i = 0; i < n_constrains; i++)
        for (auto const &coefficient : linear_system[i].coefficients)
            A.insert(i, coefficient.first) = coefficient.second;

    // init B
    VectorXd B(n_constrains);

    // fill B
    for (int i = 0; i < n_constrains; i++)
        B[i] = linear_system[i].b;

    // solve
    SparseMatrix<double> AT = A.transpose();
    B = AT * B;
    A = AT * A;
    A.makeCompressed();

    ConjugateGradient<SparseMatrix<double>> solver;
    solver.compute(A);
    VectorXd X = solver.solve(B);
    assert(solver.info() == Success);

    return X;
    //return vector<double>(X.data(), X.data() + X.size());
};
