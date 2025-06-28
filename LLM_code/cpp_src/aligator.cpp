#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>
#include<cmath>
#include<vector>
#include<iostream>

using namespace std;
namespace py = pybind11;
double prev_pred = 0;

struct expert {
    double prediction;
    double loss;
    int count;
    double weight;

    expert() {
        prediction = 0;
        loss = 0;
        count = 0;
        weight = 0;
    }
};

int init_experts(std::vector<std::vector<expert> > &pool, int n) {
    int count = 0;
    for (int k = 0; k <= floor(log2(n)); k++) {
        int stop = ((n + 1) >> k) - 1;
        if (stop < 1)
            break;
        std::vector<expert> elist;
        for (int i = 1; i <= stop; i++) {
            expert e;
            elist.push_back(e);
            if( k > 4 ) count++;
        }
        pool.push_back(elist);
    }
    return count;
}

void get_awake_set(std::vector<int> &index, int t, int n) {
    // int j = 0;
    for (int k = 0; k <= floor(log2(t)); k++) {
        int i = (t >> k);
        if (((i + 1) << k) - 1 > n  || k<=4)
            index.push_back(-1);
        // j++;
        else
            index.push_back(i);
    }
    // if(j == floor(log2(t))+1){
    //     std::cout<<"failed! "<<t<<std::endl;
    // }
}

double get_forecast(std::vector<int> &awake_set,
                    std::vector<std::vector<expert> > &pool,
                    double &normalizer, int pool_size) {
    double output = 0;
    normalizer = 0;
    int i;
    for (int k = 0; k < awake_set.size(); k++) {
        if (awake_set[k] == -1) continue;
        i = awake_set[k] - 1;
        if (pool[k][i].weight == 0) {
            pool[k][i].weight = 1.0 / pool_size;
            // added to reduce jittery output for isotonic case
            pool[k][i].prediction = prev_pred;
        }
        output = output + (pool[k][i].weight * pool[k][i].prediction);
        normalizer = normalizer + pool[k][i].weight;
    }
    return output / normalizer;
}

void compute_losses(std::vector<int> &awake_set,
                    std::vector<std::vector<expert> > &pool,
                    std::vector<double> &losses, double y,
                    double B, int n, double sigma, double delta) {
    int i;
    double norm = 2 * (B + sigma * sqrt(log(2 * n / delta))) * (B + sigma * sqrt(log(2 * n / delta)));
    // double norm = sigma;

    for (int k = 0; k < awake_set.size(); k++) {
        if (awake_set[k] == -1) {
            losses.push_back(-1);
        } else {
            i = awake_set[k] - 1;
            double loss = (y - pool[k][i].prediction) * (y - pool[k][i].prediction) / norm;
            losses.push_back(loss);
        }
    }
}

void update_weights_and_predictions(std::vector<int> &awake_set,
                                    std::vector<std::vector<expert> > &pool,
                                    std::vector<double> &losses,
                                    double normalizer, double y) {
    double norm = 0;
    int i;
    // compute new normalizer
    for (int k = 0; k < awake_set.size(); k++) {
        if (awake_set[k] == -1) continue;
        i = awake_set[k] - 1;
        norm = norm + pool[k][i].weight * exp(-losses[k]);
    }
    // update weights and predictions
    for (int k = 0; k < awake_set.size(); k++) {
        if (awake_set[k] == -1) continue;
        i = awake_set[k] - 1;
        pool[k][i].weight = pool[k][i].weight * exp(-losses[k]) * normalizer / norm;
        pool[k][i].prediction = ((pool[k][i].prediction * pool[k][i].count) + y) / (pool[k][i].count + 1);
        pool[k][i].count = pool[k][i].count + 1;
    }
}

std::vector<double> run_aligator(int n, std::vector<double> y,
                                 std::vector<int> index,
                                 double sigma,
                                 double B, double delta) {
    prev_pred = 0;
    std::vector<double> estimates(n);
    std::vector<std::vector<expert> > pool;
    int pool_size = init_experts(pool, n);
    for (int t = 0; t < n; t++) {
        double normalizer = 0;
        std::vector<int> awake_set;
        int idx = index[t];
        double y_curr = y[idx];
        get_awake_set(awake_set, idx + 1, n);
        double output = get_forecast(awake_set, pool, normalizer, pool_size);
        estimates[idx] = output;
        // if(output == 0)
        //     std::cout<<"no predict: "<<idx<<std::endl;
        // if(idx<10 ){
        //     std::cout<<"idx: "<<idx<<" output: "<<output<<std::endl;
        //     for(int k = 0; k < awake_set.size(); k++){
        //         std::cout<< awake_set[k]<<' ';
        //     }
        //     std::cout<<std::endl;
        // }
        std::vector<double> losses;
        compute_losses(awake_set, pool, losses, y_curr, B, n, sigma, delta);
        update_weights_and_predictions(awake_set, pool, losses, normalizer, y_curr);
        prev_pred = y_curr;
    }
    return estimates;
}
		    

PYBIND11_MODULE(aligator, m) {
    m.doc() = "pybind11 aligator plugin"; // optional module docstring

    m.def("run_aligator", [](int n, std::vector<double> y, std::vector<int> index, \
			     double sigma, double B, double delta) -> py::array {
	    auto v = run_aligator(n,y,index,sigma,B,delta);
	return py::array(v.size(), v.data());
	  },py::arg("n"), py::arg("index"), py::arg("y"), py::arg("sigma"),	\
	  py::arg("B"), py::arg("delta"));
}

/*
Linux:
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` aligator.cpp -o aligator`python3-config --extension-suffix`
Mac:
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` aligator.cpp -o aligator`python3-config --extension-suffix`

 */
