#pragma once
#include <ceres/ceres.h>
#include <iostream>
#include <Eigen/Dense>

// Perform smoothing by suppressing high frequency components in DCT space.
// Suppressing the highest (num_t - low_comp) components.
class DCTCost: public ceres::CostFunction
{
public:
	DCTCost(const uint num_t, const uint low_comp, const uint dim, const uint start_dim, const uint end_dim, const double weight):
		num_t_(num_t), low_comp_(low_comp), dim_(dim), start_dim_(start_dim), end_dim_(end_dim), weight_(weight)
	{
		assert(low_comp_ < num_t_);
		assert(end_dim_ > start_dim_);
		assert(end_dim_ <= dim);
		CostFunction::set_num_residuals((end_dim_ - start_dim_) * (num_t - low_comp));
		CostFunction::mutable_parameter_block_sizes()->clear();
		for (auto i = 0u; i < num_t; i++)
			CostFunction::mutable_parameter_block_sizes()->emplace_back(dim_);
	}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
private:
	const uint num_t_, low_comp_, dim_, start_dim_, end_dim_;
	const double weight_;
};

struct TemporalSmoothCost
{
public:
	TemporalSmoothCost(const uint num_t, const uint dim): num_t_(num_t), dim_(dim)
	{
		assert(num_t > 1);
		// CostFunction::set_num_residuals(dim * num_t);
		// CostFunction::mutable_parameter_block_sizes()->clear();
		// for (auto i = 0u; i < num_t; i++)
		// 	CostFunction::mutable_parameter_block_sizes()->emplace_back(dim);
	}

	template<typename T>
	bool operator()(T const* const* parameters, T* residuals) const {
		for (uint d = 0; d < dim_; d++)
		{
			for (uint t = 0u; t < num_t_; t++)
			{
				if (t == 0) residuals[d * num_t_ + t] = parameters[t][d] - parameters[t + 1][d];
				else if (t == num_t_ - 1) residuals[d * num_t_ + t] = parameters[t][d] - parameters[t - 1][d];
				else residuals[d * num_t_ + t] = parameters[t][d] + parameters[t][d] - parameters[t - 1][d] - parameters[t + 1][d];
			}
		}
		return true;
	}
private:
	const uint num_t_, dim_;
};

class TemporalSmoothCostDiff: public ceres::CostFunction
{
public:
	TemporalSmoothCostDiff(const uint num_t, const uint dim, const uint start_dim, const uint end_dim):
		num_t_(num_t), dim_(dim), start_dim_(start_dim), end_dim_(end_dim)
	{
		assert(num_t > 1);
		assert(end_dim_ > start_dim_);
		CostFunction::set_num_residuals((end_dim_ - start_dim_) * num_t);
		CostFunction::mutable_parameter_block_sizes()->clear();
		for (auto i = 0u; i < num_t; i++)
			CostFunction::mutable_parameter_block_sizes()->emplace_back(dim);
	}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

private:
	const uint num_t_, dim_, start_dim_, end_dim_;
};