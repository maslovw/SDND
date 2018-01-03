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

	normal_distribution<double> dist_x( x, std[0] );
	normal_distribution<double> dist_y( y, std[1] );
	normal_distribution<double> dist_theta( theta, std[2] );
	random_device rd;
	default_random_engine generate(rd());
	int i = 0;
	for (auto&& particle: particles) {
		particle.id = i;
		particle.x = dist_x(generate);
		particle.y = dist_y(generate);
		particle.theta = dist_theta(generate);
		particle.weight = 1.0;
		++i;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double x_mean;
	double y_mean;
	double theta_mean;
	double const yaw_rate_sigma = 0.00001;

	default_random_engine generate;

	for (auto&& particle: particles) {
		if (fabs(yaw_rate) < yaw_rate_sigma) {
			x_mean = particle.x + velocity * delta_t * cos(particle.theta);
			y_mean = particle.y + velocity * delta_t * sin(particle.theta);
			theta_mean = particle.theta;
		}
		else {
			x_mean = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			y_mean = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			theta_mean = particle.theta +yaw_rate * delta_t;
		}

		normal_distribution<double> dist_x(x_mean, std_pos[0]);
		normal_distribution<double> dist_y(y_mean, std_pos[1]);
		normal_distribution<double> dist_psi(theta_mean, std_pos[2]);

		particle.x = dist_x(generate);
		particle.y = dist_y(generate);
		particle.theta = dist_psi(generate);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	double distance;

	for (int k = 0; k < observations.size(); k++) {

		//calculate initial distance according to the first landmark item and set initial observation ID to the first landmark id
		distance = dist(observations[k].x, observations[k].y, predicted[0].x, predicted[0].y);
		observations[k].id = predicted[0].id;

		//looping through landmark items and updating current observation id if distance is smaller
		for (int i = 1; i < predicted.size(); i++) {
			if (dist(observations[k].x, observations[k].y, predicted[i].x, predicted[i].y) < distance) {
				distance = dist(observations[k].x, observations[k].y, predicted[i].x, predicted[i].y);
				observations[k].id = predicted[i].id;
			}

		}

	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	vector<LandmarkObs> landmarks_in_range;
	vector<LandmarkObs> trans_observations;

	LandmarkObs single_observation;

	for (auto&& particle : particles) {

		landmarks_in_range.clear();
		trans_observations.clear();

		//keep landmarks within sensor range
		for (const auto& landmark :map_landmarks.landmark_list) {
			if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
				landmarks_in_range.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}

		//transform observations from car to map coordinates
		for (const auto& observation : observations) {

			double t_x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
			double t_y = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);

			trans_observations.push_back(LandmarkObs{-1, t_x, t_y});
		}

		//get landmark id for each transformed observation
		dataAssociation(landmarks_in_range, trans_observations);

		//calculate weight for each particle based on assigned transformed observations and landmark coordinates
		for (const auto& trans_observation : trans_observations) {
			double lm_x;
			double lm_y;

			for (int n = 0; n < landmarks_in_range.size(); n++) {
				if (landmarks_in_range[n].id == trans_observation.id) {
					lm_x = landmarks_in_range[n].x;
					lm_y = landmarks_in_range[n].y;
				}
			}

			double dev_x = pow((trans_observation.x - lm_x),2);
			double dev_y = pow((trans_observation.y - lm_y),2);
			particle.weight *= 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exp (-(dev_x / (2 * pow(std_landmark[0],2)) + (dev_y / (2 * pow(std_landmark[1],2)))));
		}

	}
}

void ParticleFilter::resample() {
	uniform_int_distribution<int> disc_dist(0, num_particles-1);

	default_random_engine gen;
	int index = disc_dist(gen);

	vector<Particle> resamp_particles;
	Particle single_particle;

	weights.clear();

	double sum = 0.0;
	double beta = 0.0;

	//calculate normalized weight
	for (const auto& particle : particles) {
		sum += particle.weight;
	}

	for (const auto& particle : particles) {
		weights.push_back(particle.weight/sum);
	}

	double max_weight = *max_element(weights.begin(), weights.end());

	//uniform generator;
	uniform_real_distribution<double> unidistribution(0.0, max_weight * 2);

	//resampling
	for (int i = 0; i < num_particles; i++) {
		beta = beta + unidistribution(gen);

		while (weights[index] < beta) {
			beta = beta - weights[index];
			if (index == num_particles) {
				index = 0;
			} else {
				index++;
			}
		}

		single_particle.x = particles[index].x;
		single_particle.y = particles[index].y;
		single_particle.theta = particles[index].theta;
		single_particle.weight = 1.0;

		resamp_particles.push_back(single_particle);

	}

	particles = resamp_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
																		 const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);
		return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);
		return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);
		return s;
}
