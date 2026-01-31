use crfs::train::{Algorithm, Trainer};

#[test]
fn test_c1_negative_validation() {
    let mut trainer = Trainer::new(Algorithm::LBFGS);

    // c1 must be non-negative
    let result = trainer.set("c1", "-1.0");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "c1 must be non-negative");

    // c1 = 0.0 should be allowed
    assert!(trainer.set("c1", "0.0").is_ok());

    // c1 > 0.0 should be allowed
    assert!(trainer.set("c1", "1.0").is_ok());
}

#[test]
fn test_c2_negative_validation() {
    let mut trainer = Trainer::new(Algorithm::LBFGS);

    // c2 must be non-negative
    let result = trainer.set("c2", "-1.0");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "c2 must be non-negative");

    // c2 = 0.0 should be allowed
    assert!(trainer.set("c2", "0.0").is_ok());

    // c2 > 0.0 should be allowed
    assert!(trainer.set("c2", "1.0").is_ok());
}

#[test]
fn test_epsilon_validation() {
    let mut trainer = Trainer::new(Algorithm::LBFGS);

    // epsilon must be non-negative
    assert!(trainer.set("epsilon", "0.0").is_ok());

    // epsilon < 0.0 should fail
    let result = trainer.set("epsilon", "-0.001");
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "epsilon must be non-negative"
    );

    // epsilon > 0.0 should be allowed
    assert!(trainer.set("epsilon", "0.001").is_ok());
    assert!(trainer.set("epsilon", "1e-5").is_ok());
}

#[test]
fn test_invalid_parameter_values() {
    let mut trainer = Trainer::new(Algorithm::LBFGS);

    // Invalid number format
    assert!(trainer.set("c1", "not_a_number").is_err());
    assert!(trainer.set("c2", "abc").is_err());
    assert!(trainer.set("epsilon", "xyz").is_err());
    assert!(trainer.set("num_memories", "not_an_int").is_err());
}

#[test]
fn test_unknown_parameter() {
    let mut trainer = Trainer::new(Algorithm::LBFGS);

    let result = trainer.set("unknown_param", "1.0");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unknown parameter")
    );
}
