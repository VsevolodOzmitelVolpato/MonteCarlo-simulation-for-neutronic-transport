#ifndef SIMULATION_CORE_H
#define SIMULATION_CORE_H

#include <random>
#include <vector>
#include <string>

const double PI = 3.14159265358979323846;
const double ROULETTE_THRESHOLD = 0.25;
const double ROULETTE_SURVIVAL_WEIGHT = 1.0;
const unsigned int MASTER_SEED = 20260912;

enum class State {absorbed, escaped_inner, escaped_outer, escaped_axial, alive, killed_by_roulette};
enum class Mode {analog, implicit_capture};

struct MaterialData { double sigma_a; double sigma_s; double A; };

struct Vector_3{
    double x,y,z;
    Vector_3() : x(0), y(0), z(0) {}
    Vector_3(double a, double b, double c) : x(a), y(b), z(c) {}
};

struct HistoryScore {
    double absorbed = 0.0;
    double escaped_inner = 0.0;
    double escaped_outer = 0.0;
    double escaped_axial = 0.0;
    double alive = 0.0;
};

struct LocalFrame { Vector_3 u; Vector_3 v; };

class Material {
    private:
    double sigma_a;
    double sigma_s;
    double sigma_t = sigma_a + sigma_s;
    double A;
    public:
    Material(double, double, double);
    double get_sigma_s() const{return sigma_s;};
    double get_sigma_a() const{return sigma_a;};
    double get_sigma_t() const{return sigma_t;};
    double get_A() const{return A;};
};

class Particle {
    private:
    Vector_3 position;
    Vector_3 direction;
    State current_state;
    double weight = 1.0;
    public:
    Particle(Vector_3, Vector_3);
    void update_state(State stat) {current_state = stat;};
    void update_position(Vector_3 pos) {position = pos;};
    void update_direction(Vector_3 dir) {direction = dir;};
    void update_weight(double wei) {weight = wei;};
    Vector_3 get_position() const{return position;};
    Vector_3 get_direction() const{return direction;};
    State get_state() const {return current_state;};
    double get_weight() const {return weight;};
};

class Geometry {
    private:
    double inner_radius, outer_radius, height;
    public:
    Geometry(double, double, double);
    double get_inner_radius() const{return inner_radius;};
    double get_outer_radius() const{return outer_radius;};
    double get_height() const{return height;};
};

class RNG {
    private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution{0.0, 1.0};
    public:
    RNG() : gen(std::random_device{}()) {}
    explicit RNG(unsigned int seed) : gen(seed) {}
    explicit RNG(std::seed_seq &seq) : gen(seq) {}
    double get_sample() { return distribution(gen); }
};

MaterialData read_material(const std::string& filename, const std::string& material_name);

Vector_3 sum(const Vector_3 &v1, const Vector_3 &v2);
Vector_3 by_a_scalar(const Vector_3 &v1, double n);
double radius_distance(const Vector_3 &vec);
double scalar_product(const Vector_3 &v1, const Vector_3 &v2);
Vector_3 cross_product(const Vector_3 &v1, const Vector_3 &v2);
LocalFrame build_local_frame(const Vector_3 &w);

double sample_free_path(RNG &rng, const Material &mat);
void update_particle(Particle &part, double s, Geometry &geo);
bool absorption_outcome(RNG &rng, const Material &mat);
double sample_mu(RNG &rng);
double sample_mu_L(RNG &rng, const Material &mat);
double sample_phi(RNG &rng);

HistoryScore simulate_history(Particle &part, RNG &rng, const Material &mat, Geometry &geo,
                              Mode mode, long long int max_collisions,
                              std::vector<double> &collision_counts, double bin_width, int n_bin);

#endif
