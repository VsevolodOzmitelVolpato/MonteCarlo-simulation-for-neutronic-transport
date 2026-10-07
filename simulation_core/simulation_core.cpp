#include "simulation_core.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

Geometry::Geometry(double a, double b, double c) : inner_radius(a), outer_radius(b), height(c) {}
Particle::Particle(Vector_3 a, Vector_3 b) : position(a), direction(b), current_state(State::alive) {}
Material::Material(double a, double b, double c) : sigma_a (a), sigma_s (b), A(c) {}

MaterialData read_material(const std::string& filename, const std::string& material_name) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: could not open " << filename << std::endl;
        std::exit(1);
    }
    std::string name;
    double sigma_a, sigma_s, A;
    while (infile >> name >> sigma_a >> sigma_s >> A) {
        if (name == material_name) {
            return MaterialData{sigma_a, sigma_s, A};
        }
    }
    std::cerr << "Error: material '" << material_name << "' not found in " << filename << std::endl;
    std::exit(1);
}

Vector_3 sum(const Vector_3 &v1, const Vector_3 &v2){
    Vector_3 v3;
    v3.x = v1.x + v2.x;
    v3.y = v1.y + v2.y;
    v3.z = v1.z + v2.z;
    return v3;
}

Vector_3 by_a_scalar(const Vector_3 &v1, double n){
    Vector_3 v3;
    v3.x = v1.x * n;
    v3.y = v1.y * n;
    v3.z = v1.z * n;
    return v3;
}

double radius_distance(const Vector_3 &vec){
    return std::sqrt(vec.x*vec.x + vec.y*vec.y);
}

double scalar_product(const Vector_3 &v1, const Vector_3 &v2){
    return v1.x* v2.x + v1.y* v2.y +v1.z* v2.z;
}

Vector_3 cross_product(const Vector_3 &v1, const Vector_3 &v2){
    Vector_3 v3;
    v3.x = v1.y* v2.z- v2.y*v1.z;
    v3.y = v1.z*v2.x - v1.x*v2.z;
    v3.z = v1.x*v2.y - v2.x*v1.y;
    return v3;
}

LocalFrame build_local_frame(const Vector_3 &w){
    Vector_3 helper;
    if (std::fabs(w.x) > std::fabs(w.y)){
        helper = Vector_3(0, 1, 0);
    } else {
        helper = Vector_3(1, 0, 0);
    }
    Vector_3 u_unnormalized = cross_product(w, helper);
    double u_length = std::sqrt(scalar_product(u_unnormalized, u_unnormalized));
    Vector_3 u = by_a_scalar(u_unnormalized, 1.0/u_length);
    Vector_3 v = cross_product(w, u);
    return LocalFrame{u, v};
}

double sample_free_path(RNG &rng, const Material &mat){
    double xi = rng.get_sample();
    double sigma_t = mat.get_sigma_t();
    return -(std::log(xi))/sigma_t;
}

void update_particle(Particle &part, double s, Geometry &geo){
    Vector_3 old_position = part.get_position();
    part.update_position(sum(old_position, by_a_scalar(part.get_direction(), s)));
    if (std::fabs(part.get_position().z) > geo.get_height()/2 ){
        part.update_state(State::escaped_axial);}
    else if (radius_distance(part.get_position()) < geo.get_inner_radius()){
             part.update_state(State::escaped_inner);}
         else if (radius_distance(part.get_position()) > geo.get_outer_radius()){
             part.update_state(State::escaped_outer);
         }
}

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

double sample_mu_L(RNG &rng, const Material &mat){
    double mu = sample_mu(rng);
    return (mat.get_A()*mu +1)/std::sqrt(mat.get_A()*mat.get_A() + 2*mat.get_A()*mu + 1);
}

double sample_phi(RNG &rng){
    double xi = rng.get_sample();
    return xi*2*PI;
}

HistoryScore simulate_history(Particle &part, RNG &rng, const Material &mat, Geometry &geo,
                              Mode mode, long long int max_collisions,
                              std::vector<double> &collision_counts, double bin_width, int n_bin){
    HistoryScore score;
    int count = 0;
    while((part.get_state() == State::alive) && (count < max_collisions)){
         double s = sample_free_path(rng, mat);
         update_particle(part, s, geo);
         double r = radius_distance(part.get_position());
         if (part.get_state() == State::alive){
            if (r == geo.get_outer_radius()){
                collision_counts[n_bin - 1] += part.get_weight();}
            else{
                int bin = (r - geo.get_inner_radius())/bin_width;
                collision_counts[bin] += part.get_weight();}

            if (mode == Mode::analog){
                if (absorption_outcome(rng, mat)){
                    part.update_state(State::absorbed);
                    score.absorbed = 1.0;}
            }
            else{
                double w = part.get_weight();
                score.absorbed += w * mat.get_sigma_a()/mat.get_sigma_t();
                double new_weight = w * mat.get_sigma_s()/mat.get_sigma_t();
                part.update_weight(new_weight);
                if (new_weight < ROULETTE_THRESHOLD){
                    if (rng.get_sample() < new_weight/ROULETTE_SURVIVAL_WEIGHT){
                        part.update_weight(ROULETTE_SURVIVAL_WEIGHT);}
                    else{
                        part.update_state(State::killed_by_roulette);}
                }
            }

            if (part.get_state() == State::alive){
                Vector_3 current_direction = part.get_direction();
                double mu_l = sample_mu_L(rng, mat), phi = sample_phi(rng);
                LocalFrame frame = build_local_frame(current_direction);
                Vector_3 term1 = by_a_scalar(current_direction, mu_l);
                Vector_3 term2 = by_a_scalar(frame.u, std::sqrt(1 - mu_l*mu_l) * std::cos(phi));
                Vector_3 term3 = by_a_scalar(frame.v, std::sqrt(1 - mu_l*mu_l) * std::sin(phi));
                Vector_3 new_direction = sum(sum(term1, term2), term3);
                double length = std::sqrt(scalar_product(new_direction, new_direction));
                new_direction = by_a_scalar(new_direction, 1.0/length);
                part.update_direction(new_direction);}
         }
         count ++;
    };

    if (part.get_state() == State::escaped_inner){
        score.escaped_inner = part.get_weight();}
    else if (part.get_state() == State::escaped_outer){
            score.escaped_outer = part.get_weight();}
        else if (part.get_state() == State::escaped_axial){
                score.escaped_axial = part.get_weight();}
            else if (part.get_state() == State::alive){
                    score.alive = part.get_weight();}

    return score;
}
