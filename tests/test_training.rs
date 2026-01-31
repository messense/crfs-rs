use crfs::Attribute;
use crfs::train::{Algorithm, Trainer};

#[test]
fn test_basic_training() {
    // Create training data
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![],
        vec![Attribute::new("clean", 1.0)],
    ];
    let yseq = vec![
        "sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "sunny", "sunny", "rainy",
    ];

    // Create and configure trainer
    let mut trainer = Trainer::new(true);
    trainer.select(Algorithm::LBFGS).unwrap();
    trainer.append(&xseq, &yseq).unwrap();

    // Set parameters
    trainer.set("c1", "0.0").unwrap();
    trainer.set("c2", "1.0").unwrap();
    trainer.set("max_iterations", "50").unwrap();

    // Use NamedTempFile for automatic cleanup on panic
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let model_path = temp_file.path();
    let result = trainer.train(model_path);

    // Check that training completed
    match result {
        Ok(_) => {
            println!("Training completed successfully!");
            // Check that model file was created
            assert!(temp_file.path().exists());
        }
        Err(e) => {
            panic!("Training failed: {}", e);
        }
    }

    // temp_file is automatically cleaned up when it goes out of scope
}

#[test]
fn test_trainer_params() {
    let mut trainer = Trainer::new(false);

    // Test setting and getting parameters
    trainer.set("c1", "0.5").unwrap();
    trainer.set("c2", "2.0").unwrap();
    trainer.set("max_iterations", "100").unwrap();

    assert_eq!(trainer.get("c1").unwrap(), "0.5");
    // Note: Rust's f64::to_string() formats 2.0 as "2" without decimal
    // We parse back to f64 for robust comparison
    let c2_value: f64 = trainer.get("c2").unwrap().parse().unwrap();
    assert!((c2_value - 2.0).abs() < f64::EPSILON);
    assert_eq!(trainer.get("max_iterations").unwrap(), "100");
}

#[test]
fn test_trainer_validation() {
    let mut trainer = Trainer::new(false);

    // Use NamedTempFile for automatic cleanup on panic
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let model_path = temp_file.path();

    // Should fail without algorithm selection
    let result = trainer.train(model_path);
    assert!(result.is_err());

    // Should fail without training data
    trainer.select(Algorithm::LBFGS).unwrap();
    let result = trainer.train(model_path);
    assert!(result.is_err());

    // temp_file is automatically cleaned up when it goes out of scope
}
