// main.cpp
// -----------------------------------------------------------------
// Simulazione Monte Carlo del trasporto di neutroni in una lastra 1D
// STEP 1-6 di N: strutture di base (RNG, materiale, particella)
//                + campionamento del libero cammino
//                + assorbimento/diffusione e direzione isotropa
//                + storia completa di un singolo neutrone
//                + ciclo su milioni di storie, tally e statistica
//                + confronto con la teoria (limite esatto + scala
//                  fisica data dalla lunghezza di diffusione)
// -----------------------------------------------------------------

#include <iostream>
#include <random>
#include <cstdint>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------
// Generatore di numeri casuali.
// Incapsula un Mersenne Twister a 64 bit e restituisce numeri
// casuali uniformi in [0, 1). Verra' usato per campionare il
// libero cammino, il tipo di interazione e la direzione di
// diffusione (step successivi).
// ---------------------------------------------------------------
class RNG {
public:
    explicit RNG(uint64_t seed) : engine_(seed), dist_(0.0, 1.0) {}

    double uniform() {
        return dist_(engine_);
    }

private:
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> dist_;
};

// ---------------------------------------------------------------
// Materiale: sezioni d'urto macroscopiche, in cm^-1.
//   sigma_a -> probabilita' di assorbimento per unita' di percorso
//   sigma_s -> probabilita' di diffusione per unita' di percorso
//   sigma_t -> sezione d'urto totale (sigma_a + sigma_s)
// ---------------------------------------------------------------
struct Material {
    double sigma_a;
    double sigma_s;
    double sigma_t;

    Material(double abs_xs, double scat_xs)
        : sigma_a(abs_xs), sigma_s(scat_xs), sigma_t(abs_xs + scat_xs) {}

    // Probabilita' che, dato un urto, questo sia di assorbimento
    double prob_absorption() const {
        return sigma_a / sigma_t;
    }
};

// ---------------------------------------------------------------
// Stato di un neutrone.
//   x       -> posizione lungo lo spessore della lastra [cm]
//   mu      -> coseno dell'angolo tra la direzione di volo e l'asse x
//              (mu > 0: si muove verso x crescenti, mu < 0: decrescenti)
//   alive   -> false quando la storia del neutrone e' terminata
//   outcome -> come e' terminata la storia (usato dallo step 4 in poi)
// ---------------------------------------------------------------
enum class Outcome { NONE, TRANSMITTED, REFLECTED, ABSORBED };

struct Particle {
    double x;
    double mu;
    bool alive;
    Outcome outcome;

    Particle(double x0, double mu0)
        : x(x0), mu(mu0), alive(true), outcome(Outcome::NONE) {}
};

// Nome leggibile di un esito, utile per stampare i risultati
const char* outcome_name(Outcome o) {
    switch (o) {
        case Outcome::TRANSMITTED: return "TRASMESSO";
        case Outcome::REFLECTED:   return "RIFLESSO";
        case Outcome::ABSORBED:    return "ASSORBITO";
        default:                   return "NESSUNO";
    }
}

// ---------------------------------------------------------------
// Campiona la distanza percorsa dal neutrone fino al prossimo urto.
// La distanza tra due urti segue una distribuzione esponenziale
// con parametro Sigma_t: p(s) = Sigma_t * exp(-Sigma_t * s).
// Si campiona con il metodo della trasformata inversa:
//   s = -ln(xi) / Sigma_t ,   xi ~ Uniforme(0,1)
//
// Nota: usiamo xi = 1 - rng.uniform() invece di rng.uniform()
// direttamente. Il generatore restituisce numeri in [0,1) (lo 0
// e' incluso, l'1 no); se per caso xi = 0, ln(0) = -infinito e
// otterremmo un libero cammino infinito. Con 1 - xi otteniamo
// numeri in (0,1], quindi il caso critico xi = 0 non si presenta
// mai (ln(1) = 0 e' perfettamente valido).
// ---------------------------------------------------------------
double sample_free_path(const Material& mat, RNG& rng) {
    double xi = 1.0 - rng.uniform();
    return -std::log(xi) / mat.sigma_t;
}

// ---------------------------------------------------------------
// Decide se un urto e' un assorbimento oppure una diffusione.
// La probabilita' di assorbimento, dato un urto, e' Sigma_a/Sigma_t.
// Si confronta un numero casuale uniforme con questa soglia: se xi
// cade sotto la soglia e' assorbimento, altrimenti e' diffusione.
// ---------------------------------------------------------------
bool is_absorbed(const Material& mat, RNG& rng) {
    double xi = rng.uniform();
    return xi < mat.prob_absorption();
}

// ---------------------------------------------------------------
// Campiona la nuova direzione dopo una diffusione isotropa.
// "Isotropa nel laboratorio" significa che la nuova direzione e'
// distribuita uniformemente su tutta la sfera. Nel nostro problema
// 1D (lastra piana) l'angolo azimutale non conta per simmetria;
// resta solo il coseno polare mu = cos(theta). Per una direzione
// uniforme sulla sfera si dimostra che e' mu stesso (non theta) a
// essere uniforme in [-1, 1], quindi:
//   mu' = 2*xi - 1 ,   xi ~ Uniforme(0,1)
//
// Nota fisica: l'isotropia nel laboratorio e' una semplificazione.
// L'urto elastico neutrone-nucleo e' isotropo nel sistema del
// CENTRO DI MASSA, il che produce un'asimmetria in avanti nel
// laboratorio (tanto piu' marcata quanto piu' leggero e' il nucleo
// bersaglio). Per ora usiamo l'approssimazione isotropa-lab; il
// trattamento corretto e' rimandabile a un'estensione futura.
// ---------------------------------------------------------------
double sample_isotropic_direction(RNG& rng) {
    double xi = rng.uniform();
    return 2.0 * xi - 1.0;
}

// ---------------------------------------------------------------
// Segue un neutrone urto dopo urto finche' non esce dalla lastra
// (riflesso o trasmesso) oppure viene assorbito. Modifica p
// direttamente e ne restituisce l'esito finale.
//
// verbose = true stampa ogni urto (comodo per controllare a mano
// che il ciclo si comporti come ci si aspetta); di default e'
// false, cosi' nello step 5 potremo chiamarla milioni di volte
// senza stampare nulla.
// ---------------------------------------------------------------
Outcome simulate_history(Particle& p, const Material& mat, double L,
                          RNG& rng, bool verbose = false) {
    int n_collisions = 0;

    while (p.alive) {
        // 1) libero cammino e avanzamento della posizione
        double s = sample_free_path(mat, rng);
        p.x += s * p.mu;

        // 2) e' uscito dalla lastra?
        if (p.x < 0.0) {
            p.alive = false;
            p.outcome = Outcome::REFLECTED;
        } else if (p.x > L) {
            p.alive = false;
            p.outcome = Outcome::TRANSMITTED;
        }
        // 3) e' ancora dentro: urto vero, assorbimento o diffusione?
        else if (is_absorbed(mat, rng)) {
            p.alive = false;
            p.outcome = Outcome::ABSORBED;
        } else {
            p.mu = sample_isotropic_direction(rng);
            ++n_collisions;
            if (verbose) {
                std::cout << "  urto #" << n_collisions << ": x = " << p.x
                          << " cm, nuova mu = " << p.mu << "\n";
            }
        }
    }
    return p.outcome;
}

// ---------------------------------------------------------------
// Accumula i risultati di tante storie, contando quante finiscono
// in ciascuno dei tre esiti possibili.
// ---------------------------------------------------------------
struct Tally {
    long long n_transmitted = 0;
    long long n_reflected   = 0;
    long long n_absorbed    = 0;

    void record(Outcome o) {
        switch (o) {
            case Outcome::TRANSMITTED: ++n_transmitted; break;
            case Outcome::REFLECTED:   ++n_reflected;   break;
            case Outcome::ABSORBED:    ++n_absorbed;    break;
            default: break;
        }
    }
};

// ---------------------------------------------------------------
// Stima di una probabilita' e del relativo errore statistico a
// partire da un conteggio su N storie indipendenti. Ogni storia e'
// un esperimento bernoulliano ("e' successo o no"), quindi:
//   p_hat        = conteggio / N
//   sigma(p_hat) = sqrt( p_hat * (1 - p_hat) / N )
// sigma(p_hat) e' l'incertezza statistica sulla stima: si riduce
// come 1/sqrt(N), il prezzo da pagare per un metodo Monte Carlo.
// ---------------------------------------------------------------
struct Estimate {
    double value;
    double error;
};

Estimate estimate_probability(long long count, long long n) {
    double p = static_cast<double>(count) / static_cast<double>(n);
    double err = std::sqrt(p * (1.0 - p) / static_cast<double>(n));
    return {p, err};
}

int main() {
    // Geometria: lastra piana di spessore L [cm]
    const double L = 5.0;

    // Materiale di esempio (valori illustrativi, li renderemo
    // configurabili piu' avanti)
    Material mat(0.2, 0.8); // sigma_a = 0.2 cm^-1, sigma_s = 0.8 cm^-1

    // Generatore di numeri casuali con seme fisso (per riproducibilita')
    RNG rng(12345);

    // --- Step 5: la simulazione vera, su milioni di storie ---
    const long long n_histories = 10'000'000;
    Tally tally;

    auto t_start = std::chrono::steady_clock::now();
    for (long long i = 0; i < n_histories; ++i) {
        Particle neutron(0.0, 1.0); // entra perpendicolarmente da x = 0
        Outcome result = simulate_history(neutron, mat, L, rng);
        tally.record(result);
    }
    auto t_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    Estimate p_trans = estimate_probability(tally.n_transmitted, n_histories);
    Estimate p_refl  = estimate_probability(tally.n_reflected,   n_histories);
    Estimate p_abs   = estimate_probability(tally.n_absorbed,    n_histories);

    std::cout << "--- Risultati su " << n_histories << " storie simulate ---\n";
    std::cout << "L = " << L << " cm, Sigma_t = " << mat.sigma_t
              << " cm^-1 (spessore ottico L*Sigma_t = " << L * mat.sigma_t
              << " liberi cammini medi)\n\n";

    std::cout << "Trasmessi: " << tally.n_transmitted << "  ->  P = "
              << p_trans.value << " +/- " << p_trans.error << "\n";
    std::cout << "Riflessi:  " << tally.n_reflected << "  ->  P = "
              << p_refl.value << " +/- " << p_refl.error << "\n";
    std::cout << "Assorbiti: " << tally.n_absorbed << "  ->  P = "
              << p_abs.value << " +/- " << p_abs.error << "\n";
    std::cout << "\nSomma delle probabilita' (controllo, deve fare 1) = "
              << (p_trans.value + p_refl.value + p_abs.value) << "\n";

    std::cout << "\nTempo di calcolo: " << elapsed_s << " s ("
              << static_cast<long long>(n_histories / elapsed_s)
              << " storie/s)\n";

    // --- Step 6: confronto con la teoria ---

    // (1) Limite inferiore ESATTO sulla trasmissione: legge di
    // Beer-Lambert per il fascio "non collided" (nessun urto).
    // Un neutrone che raggiunge x=L senza mai urtare e' per
    // definizione trasmesso, quindi P(trasmissione) simulata deve
    // essere >= P0. Nessuna approssimazione: e' matematica esatta.
    double p_uncollided = std::exp(-mat.sigma_t * L);

    // (2) Lunghezza di diffusione, come scala fisica di riferimento
    // (NON una predizione precisa delle tre probabilita': la teoria
    // della diffusione non descrive bene un fascio collimato al
    // bordo, ma dice se la lastra e' "otticamente spessa").
    //   D      = 1 / (3 * Sigma_tr)
    //   Sigma_tr = Sigma_t per diffusione isotropa nel laboratorio
    //              (il coseno medio di diffusione e' 0, quindi
    //              nessuna correzione di trasporto e' necessaria)
    //   L_diff = sqrt(D / Sigma_a)
    double sigma_tr = mat.sigma_t;
    double D = 1.0 / (3.0 * sigma_tr);
    double diffusion_length = std::sqrt(D / mat.sigma_a);

    std::cout << "\n--- Step 6: confronto con la teoria ---\n";
    std::cout << "P(trasmissione senza urti) = exp(-Sigma_t*L) = "
              << p_uncollided << "\n";
    std::cout << "P(trasmissione) simulata                     = "
              << p_trans.value << "\n";
    std::cout << "Simulazione coerente con il limite esatto: "
              << (p_trans.value >= p_uncollided ? "SI" : "NO -> ERRORE")
              << "\n";

    std::cout << "\nD = 1/(3*Sigma_tr)      = " << D << " cm\n";
    std::cout << "L_diff = sqrt(D/Sigma_a) = " << diffusion_length << " cm\n";
    std::cout << "L / L_diff               = " << (L / diffusion_length)
              << "  (>> 1: lastra otticamente spessa, coerente con "
              << "l'assorbimento e la riflessione dominanti sulla "
              << "trasmissione diretta)\n";

    return 0;
}
