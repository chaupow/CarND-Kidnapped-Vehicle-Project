/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

const int NUM_PARTICLES = 100;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: DONE Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if (is_initialized)
  {
    return;
  }

  num_particles = NUM_PARTICLES; // TODO: Set the number of particles

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  weights.reserve(num_particles);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle new_particle;
    new_particle.id = i;
    new_particle.x = dist_x(gen);
    new_particle.y = dist_y(gen);
    new_particle.theta = dist_theta(gen);
    new_particle.weight = 1;
    particles.push_back(new_particle);

    weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: DONE Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);

  for (auto &particle : particles)
  {
    if (fabs(yaw_rate) < 0.0001) // if the car is going straightish, move the car in the direction it's heading
    {
      particle.x = particle.x + velocity * delta_t * cos(particle.theta);
      particle.y = particle.y + velocity * delta_t * sin(particle.theta);
      particle.theta = particle.theta;
    }
    else
    {
      particle.x = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta = particle.theta + delta_t * yaw_rate;
    }
    // make it fuzzy
    particle.x = particle.x + noise_x(gen);
    particle.y = particle.y + noise_y(gen);
    particle.theta = particle.theta + noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: DONE Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (auto& observation : observations)
  {
    double min_distance = std::numeric_limits<double>::max();
    int prediction_id = -1;

    // find prediction that is closest to observation in distance
    for (auto & prediction : predicted)
    { 
      double distance_x = observation.x - prediction.x;
      double distance_y = observation.y - prediction.y;
      double distance = distance_x * distance_x + distance_y * distance_y;

      if (distance < min_distance)
      {
        min_distance = distance;
        prediction_id = prediction.id;
      }
    }
    observation.id = prediction_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  weights.clear();
  double std_landmark_range = std_landmark[0];
  double std_landmark_bearing = std_landmark[1];

  for (auto &particle : particles)
  {
    double x = particle.x;
    double y = particle.y;
    double theta = particle.theta;

    vector<LandmarkObs> landmark_candidates;

    for (auto &landmark : map_landmarks.landmark_list)
    {
      float landmark_x = landmark.x_f;
      float landmark_y = landmark.y_f;
      int id = landmark.id_i;

      double dist_x = x - landmark_x;
      double dist_y = y - landmark_y;
      bool is_in_sensor_range = dist_x * dist_x + dist_y * dist_y <= sensor_range * sensor_range;
      if (is_in_sensor_range) {
        landmark_candidates.push_back(LandmarkObs{id, landmark_x, landmark_y});
      }

      vector<LandmarkObs> vehicle_view_observations;
      for (auto &observation : observations) {
        double xx = cos(theta) * observation.x - sin(theta) * observation.y + x;
        double yy = sin(theta) * observation.x + cos(theta) * observation.y + y;
        vehicle_view_observations.push_back(LandmarkObs{observation.id, xx, yy});
      }

      dataAssociation(landmark_candidates, vehicle_view_observations);

      particle.weight = 1.0;

      for (auto &vehicle_view_observation : vehicle_view_observations) {
        double landmark_x, landmark_y;
        for(auto& landmark_candidate : landmark_candidates) {
          if (landmark_candidate.id == vehicle_view_observation.id) {
            landmark_x = landmark_candidate.x;
            landmark_y = landmark_candidate.y;
          }
        }

        // Calculating weight.
        double dist_x = vehicle_view_observation.x - landmark_x;
        double dist_y = vehicle_view_observation.y - landmark_y;

        double weight = (1 / (2 * M_PI * std_landmark_range * std_landmark_bearing)) * exp(-(dist_x * dist_x / (2 * std_landmark_range * std_landmark_range) + (dist_y * dist_y / (2 * std_landmark_bearing * std_landmark_bearing))));
        if (weight == 0) { particle.weight *= 0.00001; }
        else { particle.weight *= weight; }
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Get weights and max weight.
  vector<double> weights;
  double max_weight = std::numeric_limits<double>::min();
  for (auto& particle: particles)
  {
    weights.push_back(particle.weight);
    // if (particle.weight > max_weight) { max_weight = particle.weight; }
    max_weight = fmax(max_weight, particle.weight);
  }

  // Creating distributions.
  std::uniform_real_distribution<double> dist_double(0.0, max_weight);
  std::uniform_int_distribution<int> dist_rand_particle(0, num_particles - 1);

  std::default_random_engine gen;

  // Generating index.
  int random_particle_id = dist_rand_particle(gen);

  double beta = 0.0;

  vector<Particle> resampledParticles;
  for (int i = 0; i < num_particles; i++) {
    beta += dist_double(gen) * 2.0;
    while (beta > weights[random_particle_id]) {
      beta -= weights[random_particle_id];
      random_particle_id = (random_particle_id + 1) % num_particles;
    }
    resampledParticles.push_back(particles[random_particle_id]);
  }

  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}