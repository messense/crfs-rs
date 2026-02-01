use crfs::Attribute;
use crfs::train::Trainer;

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
    let mut trainer = Trainer::lbfgs();
    trainer.verbose(true).append(&xseq, &yseq).unwrap();

    // Set parameters
    trainer.params_mut().set_c1(0.0).unwrap();
    trainer.params_mut().set_c2(1.0).unwrap();
    trainer.params_mut().set_max_iterations(50).unwrap();

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
    let mut trainer = Trainer::lbfgs();

    // Test setting and getting parameters
    trainer.params_mut().set_c1(0.5).unwrap();
    trainer.params_mut().set_c2(2.0).unwrap();
    trainer.params_mut().set_max_iterations(100).unwrap();

    assert_eq!(trainer.params().c1(), 0.5);
    assert!((trainer.params().c2() - 2.0).abs() < f64::EPSILON);
    assert_eq!(trainer.params().max_iterations(), 100);
}

#[test]
fn test_trainer_validation() {
    let mut trainer = Trainer::lbfgs();

    // Use NamedTempFile for automatic cleanup on panic
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let model_path = temp_file.path();

    // Should fail without training data
    let result = trainer.train(model_path);
    assert!(result.is_err());

    // temp_file is automatically cleaned up when it goes out of scope
}

#[test]
fn test_lbfgs_with_l1_regularization() {
    // Test OWL-QN (c1 > 0) which should use forced backtracking line search
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
    ];
    let yseq = vec!["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"];

    let mut trainer = Trainer::lbfgs();
    trainer.append(&xseq, &yseq).unwrap();

    // Enable L1 regularization (OWL-QN) - this forces backtracking line search
    trainer.params_mut().set_c1(0.1).unwrap();
    trainer.params_mut().set_c2(1.0).unwrap();
    trainer.params_mut().set_max_iterations(50).unwrap();

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let result = trainer.train(temp_file.path());

    assert!(result.is_ok(), "OWL-QN training failed: {:?}", result.err());
    assert!(temp_file.path().exists());
}

#[test]
fn test_pruned_model_roundtrip() {
    use crfs::Model;

    // Train a model that will have some zero-weight features after pruning
    let xseq = vec![
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("shop", 1.0)],
        vec![Attribute::new("walk", 1.0)],
    ];
    let yseq = vec!["sunny", "rainy", "sunny"];

    let mut trainer = Trainer::lbfgs();
    trainer.append(&xseq, &yseq).unwrap();
    trainer.params_mut().set_max_iterations(20).unwrap();

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    trainer.train(temp_file.path()).unwrap();

    // Load the pruned model and verify it can tag sequences
    let model_data = std::fs::read(temp_file.path()).expect("Failed to read model file");
    let model = Model::new(&model_data).expect("Failed to load pruned model");
    let tagger = model.tagger().expect("Failed to create tagger");

    // Tag a new sequence
    let test_xseq = vec![
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("shop", 1.0)],
    ];

    let result = tagger.tag(&test_xseq);
    assert!(
        result.is_ok(),
        "Tagging with pruned model failed: {:?}",
        result.err()
    );

    let tags = result.unwrap();
    assert_eq!(tags.len(), 2);
    // Each tag should be a valid label (sunny or rainy)
    for tag in &tags {
        assert!(
            *tag == "sunny" || *tag == "rainy",
            "Unexpected tag: {}",
            tag
        );
    }
}

#[test]
fn test_lbfgs_period_zero_disables_delta_test() {
    // Test that period=0 disables delta-based stopping
    let mut trainer = Trainer::lbfgs();

    // Should not error - period=0 is valid
    trainer.params_mut().set_period(0);
    assert_eq!(trainer.params().period(), 0);

    // Train with period=0 to ensure it works
    let xseq = vec![
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("shop", 1.0)],
    ];
    let yseq = vec!["sunny", "rainy"];

    trainer.append(&xseq, &yseq).unwrap();
    trainer.params_mut().set_max_iterations(10).unwrap();

    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let result = trainer.train(temp_file.path());
    assert!(
        result.is_ok(),
        "Training with period=0 failed: {:?}",
        result.err()
    );
}
