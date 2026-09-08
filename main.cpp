#include <iostream>
#include <random>
#include <cmath>

using namespace std;

enum class State {absorbed, transmitted, reflected, alive};

class Material {
    private:
    double sigma_a;
    double sigma_s;
    double sigma_t = sigma_a + sigma_s;

    public:
    Material(double, double);

    double get_sigma_s() const{return sigma_s;};
    double get_sigma_a() const{return sigma_a;};
    double get_sigma_t() const{return sigma_t;};
};

class Particle {
    private:
    double position;
    double direction;
    State current_state;

    public:
    Particle(double, double);
    
    void update_state(State stat) {current_state = stat;};
    void update_position(double pos) {position = pos;};
    void update_direction(double dir) {direction = dir;};
    
    double get_position() const{return position;};
    double get_direction() const{return direction;};
    State get_state() const {return current_state;};
};

class Geometry {
    private:
    double slab_thickness;
    
    public:
    Geometry(double);
    double get_slab_thickness() const{return slab_thickness;};
};

class RNG {
    private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution{0.0, 1.0};

    public:
    RNG() : gen(std::random_device{}()) {}

    explicit RNG(unsigned int seed) : gen(seed) {}  /// for debug runs

    double get_sample() { return distribution(gen); }
};

Geometry::Geometry(double a) : slab_thickness(a) {}

Particle::Particle(double a, double b) : position(a), direction(b), current_state(State::alive) {}

Material::Material(double a, double b) : sigma_a (a), sigma_s (b) {}

struct Results {
    long long int absorbed_count = 0, transmitted_count = 0, reflected_count = 0, alive_count = 0;

    double standard_error_abs, standard_error_trans, standard_error_ref, standard_error_alive;

    double final_proportion_absorbed, final_proportion_transmitted, final_proportion_reflected, final_proportion_alive;
};

double sample_free_path(RNG &rng, const Material &mat){
    double xi = rng.get_sample();
    double sigma_t = mat.get_sigma_t();
    return -(std::log(xi))/sigma_t;
}

void update_particle(Particle &part, double s, Geometry &geo){
    double old_position = part.get_position();
    part.update_position(old_position + s * part.get_direction());
    if (part.get_position() < 0 ){
        part.update_state(State::reflected);}
    else if (part.get_position() > geo.get_slab_thickness()){
             part.update_state(State::transmitted);}
};

bool absorption_outcome(RNG &rng, const Material &mat){
    double xi = rng.get_sample();
    if (xi < mat.get_sigma_a()/mat.get_sigma_t()){
        return true;}
    else{
        return false;}
}

double sample_mu(RNG &rng){
    double xi = rng.get_sample();
    return xi*2 - 1;
}

State simulate_history(Particle &part,RNG &rng, const Material &mat, Geometry &geo, long long int max_collisions){
    int count = 0;
    while((part.get_state() == State::alive) && (count < max_collisions)){
         double s = sample_free_path(rng, mat);
         update_particle(part, s, geo);
         if ((part.get_state() != State::transmitted) && (part.get_state() != State::reflected)){
            if (absorption_outcome(rng, mat)){
                part.update_state(State::absorbed);}
            else{
                part.update_direction(sample_mu(rng));}
         }
         count ++;
    };
    return part.get_state();
}

int main(){
    /*
    This program uses graphite as material, but it can be changed in order to obtain different 
    results for every material and experiment you want, moreover, you can choose how many particle
    you want for the MC simulation.          */

    long long int N_particles, max_collisions;
    double initial_position = 0, initial_direction = 1;

    ///cout << "Insert the number of particles that you want to simulate (MC method): ";
    ///cin >> N_particles;
    ///cout << endl;

    N_particles = 2000000;
    max_collisions = 2000000;

    double sigma_a_graphite = 0.00027, sigma_s_graphite = 0.38;  ///cm^-1
    double slab_thickness_graphite = 15; ///cm

    RNG rng;
    Results results;
    Material material(sigma_a_graphite, sigma_s_graphite);
    Geometry geometry(slab_thickness_graphite);

    for(long long int i = 0; i < N_particles; i++){
        State final_state;
        Particle particle(initial_position, initial_direction);

        final_state = simulate_history(particle, rng, material, geometry, max_collisions);
        switch (final_state){
        case State::absorbed:
            results.absorbed_count ++;
            break;
        case State::transmitted:
            results.transmitted_count ++;
            break;
        case State::reflected:
            results.reflected_count ++;
            break;
        case State::alive:
            results.alive_count ++;
            break;}
    };
    
    results.final_proportion_absorbed = static_cast<double> (results.absorbed_count)/N_particles;
    results.final_proportion_transmitted = static_cast<double> (results.transmitted_count)/N_particles;
    results.final_proportion_reflected = static_cast<double> (results.reflected_count)/N_particles;
    results.final_proportion_alive = static_cast<double> (results.alive_count)/N_particles;

    results.standard_error_abs = std::sqrt(results.final_proportion_absorbed*(1-results.final_proportion_absorbed)/N_particles);
    results.standard_error_trans = std::sqrt(results.final_proportion_transmitted*(1-results.final_proportion_transmitted)/N_particles);
    results.standard_error_ref = std::sqrt(results.final_proportion_reflected*(1-results.final_proportion_reflected)/N_particles);
    results.standard_error_alive = std::sqrt(results.final_proportion_alive*(1-results.final_proportion_alive)/N_particles);

    cout << "the proportion transmitted is " << results.final_proportion_transmitted << " and the mathematical count is " << std::exp(-sigma_a_graphite*slab_thickness_graphite) << endl;
    cout << "their difference is " << (results.final_proportion_transmitted - std::exp(-sigma_a_graphite*slab_thickness_graphite)) << endl;
    cout << "the reflected proportion is " << results.final_proportion_reflected << endl;

    return 0;
}
