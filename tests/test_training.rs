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

    // Train
    let model_path = std::env::temp_dir().join("test_model.crfsuite");
    let result = trainer.train(model_path.to_str().unwrap());

    // Check that training completed
    match result {
        Ok(_) => {
            println!("Training completed successfully!");
            // Check that model file was created
            assert!(model_path.exists());
        }
        Err(e) => {
            panic!("Training failed: {}", e);
        }
    }
}

#[test]
fn test_trainer_params() {
    let mut trainer = Trainer::new(false);

    // Test setting and getting parameters
    trainer.set("c1", "0.5").unwrap();
    trainer.set("c2", "2.0").unwrap();
    trainer.set("max_iterations", "100").unwrap();

    assert_eq!(trainer.get("c1").unwrap(), "0.5");
    assert_eq!(trainer.get("c2").unwrap(), "2");
    assert_eq!(trainer.get("max_iterations").unwrap(), "100");
}

#[test]
fn test_trainer_validation() {
    let mut trainer = Trainer::new(false);

    // Should fail without algorithm selection
    let model_path = std::env::temp_dir().join("test.crfsuite");
    let result = trainer.train(model_path.to_str().unwrap());
    assert!(result.is_err());

    // Should fail without training data
    trainer.select(Algorithm::LBFGS).unwrap();
    let result = trainer.train(model_path.to_str().unwrap());
    assert!(result.is_err());
}
