import sys
import numpy as np
import argparse
import cvxpy as cvx

# input is  --min-parts 4 --max-parts 32 --total-parts 256 --min-machines 1 --max-machines 8 --cores-per-machine  4

class Select_Train_Points(object):

    threshold_of_lambda = 0.9

    def __init__(self, min_parts, max_parts, total_parts,
                 min_machines, max_machines, cores_per_machine, budget=10.0,
                 interpolation=20):

        self.min_parts = min_parts
        self.max_parts = max_parts
        self.total_parts = total_parts
        self.min_machines = min_machines
        self.max_machines = max_machines
        self.cores_per_machine = cores_per_machine
        self.budget = budget
        self.interpolation = interpolation

    def _get_training_points(self):
        '''Enumerate all the training points given the params for experiment design'''
        machines_range = xrange(self.min_machines, self.max_machines + 1)

        scale_min = float(self.min_parts) / float(self.total_parts)
        scale_max = float(self.max_parts) / float(self.total_parts)
        scale_range = np.linspace(scale_min, scale_max, self.interpolation)

        for scale in scale_range:
            for machines in machines_range:
                if np.round(scale * self.total_parts) >= self.cores_per_machine * machines:
                    yield [scale, machines]

    def get_constraints(self, lambdas, points):
        '''Construct non-negative lambdas and budget constraints'''
        constraints = []
        constraints.append(0 <= lambdas)
        constraints.append(lambdas <= 1)
        constraints.append(self.get_cost(lambdas, points) <= self.budget)
        return constraints

    def get_cost(self, lambdas, data_points):
        cost = 0
        num_points = len(data_points)

        print num_points

        scale_min = float(self.min_parts) / float(self.total_parts)

        for i in xrange(0, num_points):
            scale = data_points[i][0]
            machines = data_points[i][1]
            cost = cost + (float(scale) / scale_min * 1.0 / float(machines) * lambdas[i])
        return cost

    def _get_training_points(self):
        machines_range = xrange(self.min_machines, self.max_machines + 1)

        scale_min = float(self.min_parts) / float(self.total_parts)
        scale_max = float(self.max_parts) / float(self.total_parts)
        scale_range = np.linspace(scale_min, scale_max, self.interpolation)

        for scale in scale_range:
            for machines in machines_range:
                if np.round(scale * self.total_parts) >= self.cores_per_machine * machines:
                    yield [scale, machines]

    def scale2parts(self, scale):
        return int(np.ceil(scale * self.total_parts))

    def get_covariance_matrices(self, features_arr):
        col_means = np.mean(features_arr, axis=0)

        means_inv = (1.0 / col_means)
        num_rows = features_arr.shape[0]

        print num_rows

        for i in xrange(0, num_rows):
            feature_row = features_arr[i,]
            ftf = np.outer(feature_row.transpose(), feature_row)
            yield np.diag(means_inv).transpose().dot(ftf.dot(np.diag(means_inv)))

    def get_objective(self, covariance_matrices, lambdas):
        num_points = len(covariance_matrices)
        num_dim = int(covariance_matrices[0].shape[0])

        print num_points
        print num_dim
        print int(covariance_matrices[0].shape[1])

        objective = 0
        matrix_part = np.zeros([num_dim, num_dim])
        for j in xrange(0, num_points):
            matrix_part = matrix_part + covariance_matrices[j] * lambdas[j]

        # return np.matrix.trace(np.linalg.inv(matrix_part))
        for i in xrange(0, num_dim):
            k_vec = np.zeros(num_dim)
            k_vec[i] = 1.0
            objective = objective + cvx.matrix_frac(k_vec, matrix_part)

        return objective

    def select(self):
        training_points = list(self._get_training_points())
        num_points = len(training_points)

        print num_points


        training_features = np.array([get_features([row[0], row[1]]) for row in training_points])

        covariance_matrices = list(self.get_covariance_matrices(training_features))

        lambdas = cvx.Variable(num_points)


        objective = cvx.Minimize(self.get_objective(covariance_matrices, lambdas))
        constraints = self.get_constraints(lambdas, training_points)

        #print len(constraints)
        #print constraints

        problem = cvx.Problem(objective, constraints)

        optimal_value = problem.solve()

        selected_lambda_idxs = []
        for i in range(0, num_points):
            if lambdas[i].value > self.threshold_of_lambda:
                selected_lambda_idxs.append((lambdas[i].value, i))

        print len(selected_lambda_idxs)

        sorted_by_lambda = sorted(selected_lambda_idxs, key=lambda l: l[0], reverse=True)

        print len(sorted_by_lambda)
        print sorted_by_lambda[5]

        return [(self.scale2parts(training_points[idx][0]), training_points[idx][0],
                 training_points[idx][1], l) for (l, idx) in sorted_by_lambda]


def get_features(training_point):
    scale = training_point[0]
    machines = training_point[1]
    return [1.0, float(scale) / float(machines), float(machines), np.log(machines)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select_Train_Points.')

    parser.add_argument('--min-parts', type=int, required=True)
    parser.add_argument('--max-parts', type=int, required=True)
    parser.add_argument('--total-parts', type=int, required=True)

    parser.add_argument('--min-machines', type=int, required=True)
    parser.add_argument('--max-machines', type=int, required=True)
    parser.add_argument('--cores-per-machine', type=int, required=True)

    parser.add_argument('--budget', type=float, default=10.0)
    parser.add_argument('--interpolation', type=int, default=20)

    args = parser.parse_args()

    Instance = Select_Train_Points(args.min_parts, args.max_parts, args.total_parts,
        args.min_machines, args.max_machines, args.cores_per_machine, args.budget,
        args.interpolation)

    results = Instance.select()
    print "Cores(useful), InputFraction(useful), Machines, Partitions, Weight"
    for result in results:
        print "%d, %f, %d, %d, %f" % (result[2] * args.cores_per_machine, result[1], result[2], result[0], result[3])
