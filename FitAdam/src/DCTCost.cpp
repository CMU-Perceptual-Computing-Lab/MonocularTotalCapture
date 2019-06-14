#include "DCTCost.h"
#define PI 3.14159265358979323846

bool DCTCost::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
	// E_d(B) = 1 / |num_t| || weight * DCT(input)||^2
	const double inv_sqrt_num_t = 1. / sqrt(num_t_);
	const int num_dim = end_dim_ - start_dim_;
	for (uint d = 0; d < num_dim; d++)
	{
		// compute DCT components for dim d
		for (uint k = low_comp_; k < num_t_; k++)
		{
			// compute the k-th DCT component
			auto& f_k = residuals[k - low_comp_ + d * (num_t_ - low_comp_)];
			f_k = 0;
			for (uint t = 0u; t < num_t_; t++)
			{
				// f_k += parameters[t][d] * cos(k * (t + 0.5) * PI / num_t_) * (2 - 2 * cos(k  * PI / num_t_));
				f_k += parameters[t][d + start_dim_] * cos(k * (t + 0.5) * PI / num_t_);
			}
			f_k = f_k * weight_ * inv_sqrt_num_t;
		}
	}

	if (jacobians)
	{
		for (uint t = 0u; t < num_t_; t++)
		{
			if (jacobians[t])
			{
				std::fill(jacobians[t], jacobians[t] + (num_t_ - low_comp_) * num_dim * dim_, 0.0);
				for (uint d = 0; d < num_dim; d++)
				{
					for (uint k = low_comp_; k < num_t_; k++)
					{
						// current residual is k - low_comp_ + d * (num_t_ - low_comp_)
						// jacobians[t][(k - low_comp_ + d * (num_t_ - low_comp_)) * dim_ + d] = cos(k * (t + 0.5) * PI / num_t_) * weight_ * inv_sqrt_num_t * (2 - 2 * cos(k  * PI / num_t_));
						jacobians[t][(k - low_comp_ + d * (num_t_ - low_comp_)) * dim_ + d + start_dim_] = cos(k * (t + 0.5) * PI / num_t_) * weight_ * inv_sqrt_num_t;
					}
				}
			}
		}
	}
	return true;
}

bool TemporalSmoothCostDiff::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
	// Multiply by difference matrix
	const int num_dim = end_dim_ - start_dim_;
	for (uint d = 0; d < num_dim; d++)
	{
		for (uint t = 0u; t < num_t_; t++)
		{
			if (t == 0) residuals[d * num_t_ + t] = parameters[t][d + start_dim_] - parameters[t + 1][d + start_dim_];
			else if (t == num_t_ - 1) residuals[d * num_t_ + t] = parameters[t][d + start_dim_] - parameters[t - 1][d + start_dim_];
			else residuals[d * num_t_ + t] = 2 * parameters[t][d + start_dim_] - parameters[t - 1][d + start_dim_] - parameters[t + 1][d + start_dim_];
		}
	}

	if (jacobians)
	{
		for (uint t = 0u; t < num_t_; t++)
		{
			if (jacobians[t])
			{
				std::fill(jacobians[t], jacobians[t] + num_t_ * num_dim * dim_, 0.0);
				if (t == 0)
				{
					for (uint d = 0; d < num_dim; d++)
					{	
						jacobians[t][(d * num_t_ + t) * dim_ + d + start_dim_] = 1;
						jacobians[t + 1][(d * num_t_ + t) * dim_ + d + start_dim_] = -1;
					}
				}
				else if (t == num_t_ - 1)
				{
					for (uint d = 0; d < num_dim; d++)
					{	
						jacobians[t][(d * num_t_ + t) * dim_ + d + start_dim_] = 1;
						jacobians[t - 1][(d * num_t_ + t) * dim_ + d + start_dim_] = -1;
					}
				}
				else
				{
					for (uint d = 0; d < num_dim; d++)
					{	
						jacobians[t][(d * num_t_ + t) * dim_ + d + start_dim_] = 2;
						jacobians[t - 1][(d * num_t_ + t) * dim_ + d + start_dim_] = -1;
						jacobians[t + 1][(d * num_t_ + t) * dim_ + d + start_dim_] = -1;
					}
				}
			}
		}
	}
}