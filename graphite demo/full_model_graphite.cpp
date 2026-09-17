/*
    This program uses graphite as material, with R1 = 150 cm, R2 = 200 cm and H = 1300 cm, in order to 
    simulate a real graphite reflector (analog method, 50 bins). These datas can be changed in order to obtain different 
    results for every material and experiment you want, moreover, you can choose how many particle
    you want for the MC simulation. If you want to see the other program, go to main_2 file         */

#include "simulation_core.h"
#include <chrono>
#include <omp.h>


int main(){
    
    Mode mode = Mode::analog;

    long long int N_particles, max_collisions;
    
    std::string material_name = "graphite";
    MaterialData graphite_data = read_material("C:\\Users\\sevin\\OneDrive\\Desktop\\Uni\\CppProgetti\\Fermilab 2.0\\main2 con scelta\\materials.txt", material_name);

    Material graphite(graphite_data.sigma_a, graphite_data.sigma_s, graphite_data.A);

    N_particles = 2000000;
    max_collisions = 100000;

    double inner_radius = 150, outer_radius = 200, height = 1300; ///cm

    RNG rng;
    Geometry geometry(inner_radius, outer_radius, height);
    Vector_3 initial_position(geometry.get_inner_radius(),0,0);
    Vector_3 initial_direction(1,0,0);

    int n_bins = 50;

    double bin_width = (geometry.get_outer_radius() - geometry.get_inner_radius())/n_bins;
    std::vector<double> S1(n_bins,0), S2(n_bins,0);
    std::vector<double> flux(n_bins), mean(n_bins), variance(n_bins), standard_error(n_bins);
    HistoryScore total_S1, total_S2;
    
    int num_threads = omp_get_max_threads();
    std::vector<RNG> thread_rngs;
    for(int i = 0; i < num_threads; i++){
        std::seed_seq seq{MASTER_SEED, static_cast<unsigned int>(i)};
        thread_rngs.emplace_back(seq);
    }

    std::vector<HistoryScore> thread_S1(num_threads), thread_S2(num_threads); 
    std::vector<std::vector<double>> thread_flux_S1(num_threads, vector<double>(n_bins,0));
    std::vector<std::vector<double>> thread_flux_S2(num_threads, vector<double>(n_bins,0));

    auto start_time = std::chrono::steady_clock::now();
    #pragma omp parallel for 
    for(long long int i = 0; i < N_particles; i++){
        int tid = omp_get_thread_num();
        std::vector<double> local_counts(n_bins,0);
        Particle particle(initial_position, initial_direction);

        HistoryScore score;
        score = simulate_history(particle, thread_rngs[tid], graphite, geometry, mode, max_collisions, local_counts, bin_width, n_bins);

        thread_S1[tid].absorbed += score.absorbed;             thread_S2[tid].absorbed += score.absorbed * score.absorbed;
        thread_S1[tid].escaped_inner += score.escaped_inner;   thread_S2[tid].escaped_inner += score.escaped_inner * score.escaped_inner;
        thread_S1[tid].escaped_outer += score.escaped_outer;   thread_S2[tid].escaped_outer += score.escaped_outer * score.escaped_outer;
        thread_S1[tid].escaped_axial += score.escaped_axial;   thread_S2[tid].escaped_axial += score.escaped_axial * score.escaped_axial;
        thread_S1[tid].alive += score.alive;                   thread_S2[tid].alive += score.alive * score.alive;
        
        for (size_t k = 0; k < S1.size(); k++){
            thread_flux_S1[tid][k] += local_counts[k];
            thread_flux_S2[tid][k] += local_counts[k]*local_counts[k];}
    };
    
    auto end_time = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    for(int t=0; t<num_threads; t++){
        total_S1.absorbed += thread_S1[t].absorbed;
        total_S2.absorbed += thread_S2[t].absorbed;
        total_S1.escaped_inner += thread_S1[t].escaped_inner;
        total_S2.escaped_inner += thread_S2[t].escaped_inner;
        total_S1.escaped_outer += thread_S1[t].escaped_outer;
        total_S2.escaped_outer += thread_S2[t].escaped_outer;
        total_S1.escaped_axial += thread_S1[t].escaped_axial;
        total_S2.escaped_axial += thread_S2[t].escaped_axial;
        total_S1.alive += thread_S1[t].alive;
        total_S2.alive += thread_S2[t].alive;

        for (int k = 0; k < n_bins; k++){
            S1[k] += thread_flux_S1[t][k];
            S2[k] += thread_flux_S2[t][k];}
    }

    double mean_absorbed = total_S1.absorbed / N_particles; 
    double variance_absorbed = total_S2.absorbed / N_particles - mean_absorbed * mean_absorbed;
    double se_absorbed = std::sqrt(variance_absorbed / N_particles); 
    double mean_alive = total_S1.alive / N_particles; 
    double variance_alive = total_S2.alive / N_particles - mean_alive * mean_alive;
    double se_alive = std::sqrt(variance_alive / N_particles); 
    double mean_escaped_ax = total_S1.escaped_axial / N_particles; 
    double variance_escaped_ax = total_S2.escaped_axial / N_particles - mean_escaped_ax * mean_escaped_ax;
    double se_escaped_ax = std::sqrt(variance_escaped_ax / N_particles);
    double mean_escaped_inner = total_S1.escaped_inner / N_particles; 
    double variance_escaped_inner = total_S2.escaped_inner / N_particles - mean_escaped_inner * mean_escaped_inner;
    double se_escaped_inner = std::sqrt(variance_escaped_inner / N_particles);
    double mean_escaped_outer = total_S1.escaped_outer / N_particles; 
    double variance_escaped_outer = total_S2.escaped_outer / N_particles - mean_escaped_outer * mean_escaped_outer;
    double se_escaped_outer = std::sqrt(variance_escaped_outer / N_particles); 

    for (size_t j = 0; j < S1.size(); j++){
        double r_in = geometry.get_inner_radius() + j * bin_width;
        double r_out = geometry.get_inner_radius() + (j+1) * bin_width;
        double shell_volume = PI*(r_out*r_out - r_in*r_in)*geometry.get_height();
        mean[j] = S1[j]/N_particles;
        variance[j] = S2[j]/N_particles - mean[j]*mean[j];
        flux[j] = mean[j]/(graphite.get_sigma_t()*shell_volume);
        standard_error[j] = std::sqrt(variance[j]/N_particles)/(graphite.get_sigma_t()*shell_volume);
    };

    cout << "inner escape  = " << mean_escaped_inner << " +/- " << se_escaped_inner << endl;
    cout << "absorbed      = " << mean_absorbed << " +/- " << se_absorbed << endl;
    cout << "outer escape  = " << mean_escaped_outer << " +/- " << se_escaped_outer << endl;
    cout << "axial escape  = " << mean_escaped_ax << " +/- " << se_escaped_ax << endl;

    if (mean_alive > 0){
        cout << "warning: some histories hit max_collisions (weight fraction " << mean_alive << " +/- " << se_alive << ")" << endl;
    } 
    
   double relative_error = se_escaped_outer / mean_escaped_outer;
   double FOM = 1.0 / (relative_error * relative_error * elapsed_seconds);

    std::cout << "time: " << elapsed_seconds << " s, relative error (outer): " << relative_error << ", FOM: " << FOM << std::endl;

    double balance = mean_absorbed + mean_escaped_inner + mean_escaped_outer + mean_escaped_ax + mean_alive;
    cout << "total balance (should be ~1): " << balance << std::endl;


    std::ofstream outfile("bin_fluc_unc.csv");
    outfile << "bin center" << "," << "flux" << "," << "uncertainty" << endl;
    for (size_t i = 0; i < flux.size(); i++ ){
        outfile << geometry.get_inner_radius() + (i+0.5)*bin_width << "," << flux[i] << "," << standard_error[i] << endl;
    };

    return 0;
}