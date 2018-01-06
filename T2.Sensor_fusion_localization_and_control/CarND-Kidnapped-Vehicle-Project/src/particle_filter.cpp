/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	weights.resize(num_particles);
	particles.resize(num_particles);

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	random_device rd;
	default_random_engine generate(rd());
	int i = 0;
	for (auto&& particle : particles) {
		particle.id = i;
		particle.x = dist_x(generate);
		particle.y = dist_y(generate);
		particle.theta = dist_theta(generate);
		particle.weight = 1.0;
		++i;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
																double velocity, double yaw_rate) {
	//Normal distribution (Gaussian)
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	default_random_engine generate;
	//Iterate through particles, checking whether yaw_rate is zero
	for (auto&& particle : particles) {
		if (abs(yaw_rate) == 0) {
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		} else {
			particle.x += (velocity / yaw_rate)
					* (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
			particle.y += (velocity / yaw_rate)
					* (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
			particle.theta += yaw_rate * delta_t;
		}
		particle.x += dist_x(generate);
		particle.y += dist_y(generate);
		particle.theta += dist_theta(generate);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
																		 std::vector<LandmarkObs>& observations) {
	double distance;

	for (int k = 0; k < observations.size(); k++) {

		//calculate initial distance according to the first landmark item and set initial observation ID to the first landmark id
		distance = 0xffffffff;

		//looping through landmark items and updating current observation id if distance is smaller
		for (int i = 0; i < predicted.size(); i++) {
			if (dist(observations[k].x, observations[k].y, predicted[i].x,
							 predicted[i].y) < distance) {
				distance = dist(observations[k].x, observations[k].y, predicted[i].x,
												predicted[i].y);
				observations[k].id = predicted[i].id;
			}

		}

	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 const std::vector<LandmarkObs> &observations,
																	 const Map &map_landmarks) {

	const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	// Run through each particle
	auto&& weight_itr = weights.begin();
	for (auto&& particle : particles) {

		// Variable to hold multi variate gaussian distribution.
		double gaussDist = 1.0;

		// For each observation
		for (auto&& observation : observations) {

			// Vehicle to map coordinates
			double trans_obs_x = observation.x * cos(particle.theta)
					- observation.y * sin(particle.theta) + particle.x;
			double trans_obs_y = observation.x * sin(particle.theta)
					+ observation.y * cos(particle.theta) + particle.y;

			// Find nearest landmark
			vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
			vector<double> landmark_obs_dist(landmarks.size());
			//Iterate through landmarks
			auto&& landmark_obs_dist_itr = landmark_obs_dist.begin();
			for (auto&& landmark : landmarks) {

				//Check to see if particle landmark is within sensor range, and calculate distance between particle and landmark.
				double landmark_part_dist = sqrt(
						pow(particle.x - landmark.x_f, 2)
								+ pow(particle.y - landmark.y_f, 2));
				if (landmark_part_dist <= sensor_range) {
					*landmark_obs_dist_itr = sqrt(
							pow(trans_obs_x - landmark.x_f, 2)
									+ pow(trans_obs_y - landmark.y_f, 2));

				} else {
					*landmark_obs_dist_itr = 999999.0;  //Fill with large number so algorithm doesn't think they are 0 and close.
				}
				++landmark_obs_dist_itr;

			}

			// Associate the observation point with its nearest landmark neighbor
			int min_pos = distance(
					landmark_obs_dist.begin(),
					min_element(landmark_obs_dist.begin(), landmark_obs_dist.end()));
			float nn_x = landmarks[min_pos].x_f;
			float nn_y = landmarks[min_pos].y_f;

			// Calculate multi-variate Gaussian distribution
			double x_diff = trans_obs_x - nn_x;
			double y_diff = trans_obs_y - nn_y;
			double b = ((x_diff * x_diff) / (2 * std_landmark[0] * std_landmark[0]))
					+ ((y_diff * y_diff) / (2 * std_landmark[1] * std_landmark[1]));
			gaussDist *= a * exp(-b);

		}

		// Update particle weights with combined multi-variate Gaussian distribution
		particle.weight = gaussDist;
		*weight_itr = particle.weight;
		++weight_itr;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles(num_particles);

	// Use discrete distribution to return particles by weight
	random_device rd;
	default_random_engine gen(rd());
	for (int i = 0; i < num_particles; ++i) {
		discrete_distribution<int> index(weights.begin(), weights.end());
		new_particles[i] = particles[index(gen)];

	}

	// Replace old particles with the resampled particles
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle,
																				 const std::vector<int>& associations,
																				 const std::vector<double>& sense_x,
																				 const std::vector<double>& sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);
	return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);
	return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);
	return s;
}
