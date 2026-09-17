#include <iostream>
#include <random>
#include <cmath>
#include <fstream>

using namespace std;

class Material {
    private:
    double sigma_a;
    double sigma_s;
    double sigma_t = sigma_a + sigma_s;
    double A = 1;

    public:
    Material(double, double);

    double get_sigma_s() const{return sigma_s;};
    double get_sigma_a() const{return sigma_a;};
    double get_sigma_t() const{return sigma_t;};
    double get_A() const{return A;};
};

Material::Material(double a, double b) : sigma_a (a), sigma_s (b) {}

class RNG {
    private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution{0.0, 1.0};

    public:
    RNG() : gen(std::random_device{}()) {}

    explicit RNG(unsigned int seed) : gen(seed) {}  /// for debug runs

    double get_sample() { return distribution(gen); }
};


double sample_mu(RNG &rng){
    double xi = rng.get_sample();
    return xi*2 - 1;
}

double sample_mu_L(RNG &rng, const Material &mat){
    double mu = sample_mu(rng);
    return (mat.get_A()*mu +1)/std::sqrt(mat.get_A()*mat.get_A() + 2*mat.get_A()*mu + 1);
}

void main_loop(){
    RNG rng;
    Material hydrogen(1,1);
    int max_iterations = 1000000;
    std::ofstream outfile("mu_lab_samples.txt");
    outfile << "mu lab values" << endl;
    for (size_t i = 0; i < max_iterations; i++){
        double mu = sample_mu_L(rng, hydrogen);
        outfile << mu << endl;
    };
}

int main(){

    main_loop();

    return 0;
}