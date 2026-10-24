# Observation time; header.frame_id is the camera optical frame.
std_msgs/Header header
string board_id
string registration_id
string calibration_revision
# valid | invalid | unconfigured
string status
bool calibration_valid
bool tracking_valid
bool board_unmoved
string board_frame
# Board pose expressed in header.frame_id, valid only when has_pose is true.
bool has_pose
geometry_msgs/Pose board_in_camera
# Zero time means unset, never unlimited validity.
builtin_interfaces/Time valid_until
bool has_reprojection_error
float64 reprojection_error_px
# Unknown height stays unset via this presence flag.
bool has_board_height
float64 board_height_m
string failure_reason
