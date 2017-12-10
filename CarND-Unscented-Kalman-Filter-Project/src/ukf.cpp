#include "ukf.h"
#include "Eigen/Dense"
#include "cmath"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF(
    int state_dim,
    int aug_state_dim,
    bool use_laser,
    bool use_radar)
:
  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_(false),
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_(use_laser),
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_(use_radar),
  ///* State dimension
  n_x_(state_dim),
  ///* Augmented state dimension
  n_aug_(aug_state_dim),
  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_(VectorXd(n_x_)),
  ///* state covariance matrix
  P_(MatrixXd(n_x_, n_x_)),
  ///* number of sigma points
  n_sigma_pts_(2 * n_aug_ +1),
  ///* predicted sigma points matrix
  Xsig_pred_(MatrixXd(n_x_, n_sigma_pts_)),
  ///* time when the state is true, in us
  time_us_(0.0),
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_square(2 * 2),
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_square(0.3 * 0.3),
  // Laser measurement noise standard deviation position1 in m
  std_laspx_square(0.15 * 0.15),
  // Laser measurement noise standard deviation position2 in m
  std_laspy_square(0.15 * 0.15),
  // Radar measurement noise standard deviation radius in m
  std_radr_square(0.3 * 0.3),
  // Radar measurement noise standard deviation angle in rad
  std_radphi_square(0.03 * 0.03),
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_square(0.3 * 0.3),
  ///* Weights of sigma points
  weights_(VectorXd(n_sigma_pts_)),
  ///* Sigma point spreading parameter
  lambda_(3 - n_x_),
  // the current NIS for radar
  NIS_radar_(0.0),
  // the current NIS for laser
  NIS_laser_(0.0),
  //create augmented state covariance
  P_aug_(MatrixXd(n_aug_, n_aug_)),
  //create sigma point matrix
  Xsig_aug_(MatrixXd(n_aug_, n_sigma_pts_))
{
}
/*
 *  Angle normalization
 */
void UKF::NormalizeAngle(double &angle) {
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0)
    angle += 2*M_PI;
  angle -= M_PI;
}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {
    /**
    Initialize state.
    */

    // first measurement
    x_ << 1, 1, 1, 1, 0.1;

    // init covariance matrix
    P_ << 0.15,    0, 0, 0, 0,
            0, 0.15, 0, 0, 0,
            0,    0, 1, 0, 0,
            0,    0, 0, 1, 0,
            0,    0, 0, 0, 1;

    // init timestamp
    time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {

      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      const float ro = meas_package.raw_measurements_(0);
      const float phi = meas_package.raw_measurements_(1);
      const float ro_dot = meas_package.raw_measurements_(2);
      x_(0) = ro * cos(phi);
      x_(1) = ro * sin(phi);
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  // Prediction
  //compute the time elapsed between the current and previous measurements
  const float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}

void UKF::PredictSigmaPoints(const double delta_t) {
  const double half_delta_t_square = 0.5 * delta_t * delta_t;

  for (int i = 0; i < n_sigma_pts_; i++) {
    const double p_x = Xsig_aug_(0, i);
    const double p_y = Xsig_aug_(1, i);
    const double v = Xsig_aug_(2, i);
    const double yaw = Xsig_aug_(3, i);
    const double yawd = Xsig_aug_(4, i);
    const double nu_a = Xsig_aug_(5, i);
    const double nu_yawdd = Xsig_aug_(6, i);
    //predicted state values
    double px_p, py_p;
    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    //add noise
    px_p += nu_a * half_delta_t_square * cos(yaw);
    py_p += nu_a * half_delta_t_square * sin(yaw);
    v_p += nu_a * delta_t;
    yaw_p += nu_yawdd * half_delta_t_square;
    yawd_p += nu_yawdd * delta_t;
    //write predicted sigma point
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
  void UKF::Prediction(const double delta_t) {

  /*****************************************************************************
  *  Generate Sigma Points
  ****************************************************************************/
  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //set lambda for non-augmented sigma points
  lambda_ = 3 - n_x_;

  //set first column of sigma point matrix
  Xsig.col(0) = x_;

  const auto sqrt_lambda_nx = sqrt(lambda_ + n_x_);
  //set remaining sigma points
  for(int i = 0; i < n_x_; i++) {
    Xsig.col(i + 1) = x_ + sqrt_lambda_nx * A.col(i);
    Xsig.col(i + 1 + n_x_) = x_ - sqrt_lambda_nx * A.col(i);
  }

  /*****************************************************************************
  *  Augment Sigma Points
  ****************************************************************************/
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);


  //set lambda for augmented sigma points
  lambda_ = 3 - n_aug_;

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_square;
  P_aug_(6, 6) = std_yawdd_square;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points
  const auto sqrt_labda_n_aug = sqrt(lambda_ + n_aug_);
  Xsig_aug_.col(0) = x_aug;
  for(int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i + 1) = x_aug + sqrt_labda_n_aug * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt_labda_n_aug * L.col(i);
  }

  /*****************************************************************************
  *  Predict Sigma Points
  ****************************************************************************/
  PredictSigmaPoints(delta_t);

  // set weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < n_sigma_pts_; i++) {  //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;

  // set measurement dimension
  const int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_pts_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_pts_; i++) {
    const double p_x = Xsig_pred_(0, i);
    const double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_square, 0,
          0, std_laspy_square;
  S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  /*****************************************************************************
    *  UKF Update for Lidar
    ****************************************************************************/
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //calculate NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // get z
  VectorXd z = meas_package.raw_measurements_;

  // set measurement dimension
  const int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_pts_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_pts_; i++) {  //2n+1 simga points

    const double p_x = Xsig_pred_(0, i);
    const double p_y = Xsig_pred_(1, i);
    const double v   = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);

    const double v1 = cos(yaw)*v;
    const double v2 = sin(yaw)*v;

    // measurement model
    const double r = sqrt(p_x*p_x + p_y*p_y);
    const double phi = ::std::atan2(p_y, p_x);
    const double r_dot = (p_x*v1 + p_y*v2) / r;
    Zsig(0, i) = r;
    Zsig(1, i) = phi;
    Zsig(2, i) = r_dot;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<    std_radr_square, 0,                 0,
          0,               std_radphi_square, 0,
          0,               0,                 std_radrd_square;
  S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);


  /*****************************************************************************
    *  UKF Update for Radar
    ****************************************************************************/
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_pts_; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  NormalizeAngle(z_diff(1));

  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}
