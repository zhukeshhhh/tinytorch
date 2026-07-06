#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <filesystem>

#include "tinytorch/core/tensor.hpp"


struct IrisSample {
    std::vector<float> features;
    int label;
};

static int species_to_label(const std::string& s) {
    if (s == "Iris-setosa")     return 0;
    if (s == "Iris-versicolor") return 1;
    if (s == "Iris-virginica")  return 2;
    throw std::runtime_error("Unknown species label: " + s);
}

static std::vector<IrisSample> load_iris(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Could not open iris csv at: " + path);

    std::vector<IrisSample> samples;
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);
        if (cells.size() < 6) continue;

        IrisSample sample;
        sample.features = {
            std::stof(cells[1]), // SepalLengthCm
            std::stof(cells[2]), // SepalWidthCm
            std::stof(cells[3]), // PetalLengthCm
            std::stof(cells[4])  // PetalWidthCm
        };
        sample.label = species_to_label(cells[5]);
        samples.push_back(std::move(sample));
    }
    return samples;
}

static std::string resolve_iris_path() {
    std::vector<std::filesystem::path> candidates = {
        "data/iris.csv",
        "../data/iris.csv",
        "../../data/iris.csv",
        "../../../data/iris.csv"
    };

    for (const auto& c : candidates)
        if (std::filesystem::exists(c)) return c.string();

    std::string tried;
    for (const auto& c : candidates) tried += "  " + c.string() + "\n";
    throw std::runtime_error(
        "Could not find iris.csv. Tried, relative to cwd (" +
        std::filesystem::current_path().string() + "):\n" + tried);
}

static void standardize(std::vector<IrisSample>& train, std::vector<IrisSample>& test) {
    const std::size_t n_features = train[0].features.size();
    std::vector<float> mean(n_features, 0.0f), stdv(n_features, 0.0f);

    for (auto& s : train)
        for (std::size_t j = 0; j < n_features; j++)
            mean[j] += s.features[j];
    for (auto& m : mean) m /= static_cast<float>(train.size());

    for (auto& s : train)
        for (std::size_t j = 0; j < n_features; j++)
            stdv[j] += (s.features[j] - mean[j]) * (s.features[j] - mean[j]);
    for (auto& v : stdv) v = std::sqrt(v / static_cast<float>(train.size())) + 1e-8f;

    for (auto& s : train)
        for (std::size_t j = 0; j < n_features; j++)
            s.features[j] = (s.features[j] - mean[j]) / stdv[j];

    for (auto& s : test)
        for (std::size_t j = 0; j < n_features; j++)
            s.features[j] = (s.features[j] - mean[j]) / stdv[j];
}


static std::shared_ptr<Tensor> init_weight(
    std::size_t rows, std::size_t cols, Device device,
    std::mt19937& gen, const std::string& label)
{
    float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-limit, limit);

    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    for (auto& row : data)
        for (auto& v : row)
            v = dist(gen);

    return Tensor::from_vector_2d(data, device, true, label);
}

static std::shared_ptr<Tensor> init_bias(std::size_t size, Device device, const std::string& label) {
    std::vector<float> data(size, 0.0f);
    return Tensor::from_vector_1d(data, device, true, label);
}

int main() {
  try {
    const Device device = Device::CPU;

    std::vector<IrisSample> all_samples = load_iris(resolve_iris_path());

    std::mt19937 gen(42);
    std::shuffle(all_samples.begin(), all_samples.end(), gen);

    const std::size_t n_train = 120;
    std::vector<IrisSample> train_data(all_samples.begin(), all_samples.begin() + n_train);
    std::vector<IrisSample> test_data(all_samples.begin() + n_train, all_samples.end());

    standardize(train_data, test_data);

    const std::size_t input_dim  = 4;
    const std::size_t hidden_dim = 10;
    const std::size_t output_dim = 3;
    float learning_rate = 0.05f;
    const std::size_t epochs = 200;
    const float target_batch_size = 16.0f;

    auto W1 = init_weight(input_dim, hidden_dim, device, gen, "W1");
    auto b1 = init_bias(hidden_dim, device, "b1");
    auto W2 = init_weight(hidden_dim, output_dim, device, gen, "W2");
    auto b2 = init_bias(output_dim, device, "b2");

    std::vector<std::shared_ptr<Tensor>> params{W1, b1, W2, b2};

    std::vector<std::size_t> indices(train_data.size());
    for (std::size_t i = 0; i < indices.size(); i++) indices[i] = i;

    std::cout << "Training on " << train_data.size() << " samples ("
              << (device == Device::CUDA ? "CUDA" : "CPU") << ")...\n";

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.0f;
        float curr_batch_count = 0.0f;

        for (std::size_t idx : indices) {
            const auto& sample = train_data[idx];
            curr_batch_count++;

            auto x = Tensor::from_vector_1d(sample.features, device, false, "x");

            std::vector<float> target_vec(output_dim, 0.0f);
            target_vec[sample.label] = 1.0f;
            auto target = Tensor::from_vector_1d(target_vec, device, false, "target");

            std::shared_ptr<Tensor> h1      = *x * W1;
            std::shared_ptr<Tensor> h1_b    = *h1 + b1;
            std::shared_ptr<Tensor> h1_relu = h1_b->relu();
            std::shared_ptr<Tensor> h2      = *h1_relu * W2;
            std::shared_ptr<Tensor> logits  = *h2 + b2;


            std::shared_ptr<Tensor> probs        = logits->softmax();
            std::shared_ptr<Tensor> log_probs    = probs->log();
            std::shared_ptr<Tensor> dot_log_prob = log_probs->mul(target);
            std::shared_ptr<Tensor> loss         = dot_log_prob->neg();


            epoch_loss += loss->sum();

            loss->backward();

            if (curr_batch_count >= target_batch_size) {
                for (auto& p : params) {
                    p->sdg_step(learning_rate, curr_batch_count);
                    p->zero_grad();
                }
                curr_batch_count = 0.0f;
            }
        }

        if (curr_batch_count > 0.0f) {
            for (auto& p : params) {
                p->sdg_step(learning_rate, curr_batch_count);
                p->zero_grad();
            }
        }

        if ((epoch + 1) % 20 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - avg loss: " << (epoch_loss / static_cast<float>(train_data.size()))
                      << "\n";
        }
    }

    std::cout << "Training complete.\n";

    std::cout << "\nEvaluating on " << test_data.size() << " test samples...\n";
    float correct = 0.0f;

    for (auto& sample : test_data) {
        auto x = Tensor::from_vector_1d(sample.features, device, false, "x");

        std::shared_ptr<Tensor> h1      = *x * W1;
        std::shared_ptr<Tensor> h1_b    = *h1 + b1;
        std::shared_ptr<Tensor> h1_relu = h1_b->relu();
        std::shared_ptr<Tensor> h2      = *h1_relu * W2;
        std::shared_ptr<Tensor> logits  = *h2 + b2;


        std::size_t predicted = logits->argmax()[0];

        if (static_cast<int>(predicted) == sample.label) correct++;
    }

    float accuracy = 100.0f * correct / static_cast<float>(test_data.size());
    std::cout << "Test accuracy: " << accuracy << "%\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
}