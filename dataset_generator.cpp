#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <filesystem>   
#include <thread>
#include <mutex>
#include <iomanip>      

std::mutex cout_mutex;

const long long MAX_UNIVERSE_FOR_BITSET = 200000000LL; 

struct Config {
    std::string distribution_type;
    long long element_universe;
    long long num_required_elements;

    int num_clusters = 0;
    double cluster_fill_ratio = 0.9; 
};

void generateDataset(const Config& config) {
    std::string dataset_dir = "revised_datasets";
    std::string output_filename = config.distribution_type + "-" + 
                                  std::to_string(config.element_universe) + "-" + 
                                  std::to_string(config.num_required_elements) + ".txt";

    try {
        if (!std::filesystem::exists(dataset_dir)) {
            std::filesystem::create_directories(dataset_dir);
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error creating directory " << dataset_dir << ": " << e.what() << std::endl;
        return;
    }

    std::mt19937 local_gen(std::hash<std::thread::id>{}(std::this_thread::get_id()));

    long long lower_bound = 0;
    long long upper_bound = config.element_universe - 1;

    bool use_bitset = (config.element_universe <= MAX_UNIVERSE_FOR_BITSET);
    std::vector<bool> seen_elements_bitset;
    std::unordered_set<long long> unique_integers_set;

    long long collected_count = 0;

    if (use_bitset) {
        seen_elements_bitset.resize(config.element_universe, false);
    } else {
        unique_integers_set.reserve(config.num_required_elements);
    }

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Thread " << std::this_thread::get_id() << ": Starting generation for " << output_filename << "\n"
                  << "  - Universe: " << config.element_universe 
                  << ", Required Elements: " << config.num_required_elements << "\n"
                  << "  - Using " << (use_bitset ? "Bitset" : "Unordered_set") << " for uniqueness tracking.\n" << std::flush;
    }

    if (config.distribution_type == "Uniform") {
        std::uniform_int_distribution<long long> dist(lower_bound, upper_bound);

        while (collected_count < config.num_required_elements) {
            long long new_integer = dist(local_gen);
            if (use_bitset) {
                if (!seen_elements_bitset[new_integer]) {
                    seen_elements_bitset[new_integer] = true;
                    collected_count++;
                }
            } else {
                if (unique_integers_set.insert(new_integer).second) {
                    collected_count++;
                }
            }
        }
    } else if (config.distribution_type == "Clustered") {

        long long num_in_clusters = static_cast<long long>(config.num_required_elements * config.cluster_fill_ratio);
        long long num_as_noise = config.num_required_elements - num_in_clusters;

        long long cluster_region_size = config.element_universe / (config.num_clusters * 5); 
        std::vector<std::pair<long long, long long>> cluster_bounds;

        std::uniform_int_distribution<long long> cluster_start_dist(lower_bound, upper_bound - cluster_region_size);

        for (int i = 0; i < config.num_clusters; ++i) {
            long long start = cluster_start_dist(local_gen);
            cluster_bounds.push_back({start, start + cluster_region_size});
        }

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread " << std::this_thread::get_id() << ": Defined " << config.num_clusters 
                      << " clusters. Target " << num_in_clusters << " elements in clusters, "
                      << num_as_noise << " as noise.\n" << std::flush;
        }

        std::uniform_int_distribution<int> cluster_chooser(0, config.num_clusters - 1);
        while (collected_count < num_in_clusters) {
            int chosen_cluster_idx = cluster_chooser(local_gen);
            std::uniform_int_distribution<long long> element_dist(cluster_bounds[chosen_cluster_idx].first, cluster_bounds[chosen_cluster_idx].second);
            long long new_integer = element_dist(local_gen);

            if (use_bitset) {
                if (!seen_elements_bitset[new_integer]) {
                    seen_elements_bitset[new_integer] = true;
                    collected_count++;
                }
            } else {
                if (unique_integers_set.insert(new_integer).second) {
                    collected_count++;
                }
            }
        }

        std::uniform_int_distribution<long long> noise_dist(lower_bound, upper_bound);
        while (collected_count < config.num_required_elements) {
            long long new_integer = noise_dist(local_gen);

            bool is_in_cluster = false;
            for(const auto& bounds : cluster_bounds) {
                if (new_integer >= bounds.first && new_integer <= bounds.second) {
                    is_in_cluster = true;
                    break;
                }
            }
            if (is_in_cluster) continue; 

            if (use_bitset) {
                if (!seen_elements_bitset[new_integer]) {
                    seen_elements_bitset[new_integer] = true;
                    collected_count++;
                }
            } else {
                if (unique_integers_set.insert(new_integer).second) {
                    collected_count++;
                }
            }
        }
    }

    std::vector<long long> final_unique_integers;
    final_unique_integers.reserve(config.num_required_elements);
    if (use_bitset) {
        for (long long i = 0; i < config.element_universe; ++i) {
            if (seen_elements_bitset[i]) {
                final_unique_integers.push_back(i);
            }
        }
    } else {
        final_unique_integers.assign(unique_integers_set.begin(), unique_integers_set.end());
    }

    std::sort(final_unique_integers.begin(), final_unique_integers.end());

    std::ofstream outfile(std::filesystem::path(dataset_dir) / output_filename);
    if (outfile.is_open()) {
        for (long long num : final_unique_integers) {
            outfile << num << "\n";
        }
        outfile.close();
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread " << std::this_thread::get_id() << ": Finished and saved " << output_filename << "\n" << std::flush;
        }
    } else {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error: Unable to open file " << output_filename << " for writing.\n" << std::flush;
    }
}

int main() {

    std::vector<Config> scenarios = {

        {"Uniform", 100000000LL,  1000000LL},  
        {"Uniform", 100000000LL,  10000000LL}, 
        {"Uniform", 1000000000LL, 1000000LL},  
        {"Uniform", 1000000000LL, 10000000LL}, 

        {"Clustered", 100000000LL,  1000000LL,  10}, 
        {"Clustered", 100000000LL,  10000000LL, 10}, 

        {"Clustered", 1000000000LL, 1000000LL,  50}, 
        {"Clustered", 1000000000LL, 10000000LL, 50}  
    };

    std::vector<std::thread> threads;
    unsigned int max_threads = std::thread::hardware_concurrency();
    std::cout << "Starting dataset generation using up to " << max_threads << " concurrent threads.\n" << std::endl;

    for (const auto& scenario : scenarios) {
        threads.emplace_back(generateDataset, scenario);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "\nAll dataset generation tasks are complete. Files are in the 'revised_datasets' folder." << std::endl;

    return 0;
}
