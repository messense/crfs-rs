use crfs::train::Trainer;

#[test]
fn test_c1_negative_validation() {
    let mut trainer = Trainer::lbfgs();

    // c1 must be non-negative
    let result = trainer.params_mut().set_c1(-1.0);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "c1 must be non-negative");

    // c1 = 0.0 should be allowed
    assert!(trainer.params_mut().set_c1(0.0).is_ok());

    // c1 > 0.0 should be allowed
    assert!(trainer.params_mut().set_c1(1.0).is_ok());
}

#[test]
fn test_c2_negative_validation() {
    let mut trainer = Trainer::lbfgs();

    // c2 must be non-negative
    let result = trainer.params_mut().set_c2(-1.0);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "c2 must be non-negative");

    // c2 = 0.0 should be allowed
    assert!(trainer.params_mut().set_c2(0.0).is_ok());

    // c2 > 0.0 should be allowed
    assert!(trainer.params_mut().set_c2(1.0).is_ok());
}

#[test]
fn test_epsilon_validation() {
    let mut trainer = Trainer::lbfgs();

    // epsilon must be non-negative
    assert!(trainer.params_mut().set_epsilon(0.0).is_ok());

    // epsilon < 0.0 should fail
    let result = trainer.params_mut().set_epsilon(-0.001);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "epsilon must be non-negative"
    );

    // epsilon > 0.0 should be allowed
    assert!(trainer.params_mut().set_epsilon(0.001).is_ok());
    assert!(trainer.params_mut().set_epsilon(1e-5).is_ok());
}

#[test]
fn test_invalid_parameter_values() {
    let mut trainer = Trainer::lbfgs();

    assert!(trainer.params_mut().set_num_memories(0).is_err());
    assert!(trainer.params_mut().set_max_iterations(0).is_err());

    let mut l2sgd_trainer = Trainer::l2sgd();
    assert!(l2sgd_trainer.params_mut().set_period(0).is_err());
    assert!(l2sgd_trainer.params_mut().set_delta(0.0).is_err());
}
