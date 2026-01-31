use crfs::train::{Algorithm, Trainer};
use crfs::{Attribute, Model};

#[test]
fn test_train_save_load_predict() {
    // Create training data - simple weather prediction
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

    // Train model
    let mut trainer = Trainer::new(false); // quiet mode
    trainer.select(Algorithm::LBFGS).unwrap();
    trainer.append(&xseq, &yseq).unwrap();
    trainer.set("c1", "0.0").unwrap();
    trainer.set("c2", "1.0").unwrap();
    trainer.set("max_iterations", "100").unwrap();

    // Use NamedTempFile for automatic cleanup on panic
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let model_path = temp_file.path().to_str().unwrap();
    trainer.train(model_path).unwrap();

    // Verify model file exists
    assert!(temp_file.path().exists());

    // Load model
    let model_data = std::fs::read(temp_file.path()).unwrap();
    let model = Model::new(&model_data).unwrap();

    // Verify model metadata
    assert_eq!(model.num_labels(), 2);
    assert_eq!(model.num_attrs(), 3);

    // Verify label mappings
    assert_eq!(model.to_label_id("sunny"), Some(0));
    assert_eq!(model.to_label_id("rainy"), Some(1));
    assert_eq!(model.to_label(0), Some("sunny"));
    assert_eq!(model.to_label(1), Some("rainy"));

    // Verify attribute mappings
    assert!(model.to_attr_id("walk").is_some());
    assert!(model.to_attr_id("shop").is_some());
    assert!(model.to_attr_id("clean").is_some());

    // Create tagger
    let mut tagger = model.tagger().unwrap();

    // Test prediction on new data
    let test_seq = vec![
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("shop", 1.0)],
        vec![Attribute::new("clean", 1.0)],
    ];

    let result = tagger.tag(&test_seq).unwrap();
    assert_eq!(result.len(), 3);

    // Verify predictions are valid labels
    for label in &result {
        assert!(*label == "sunny" || *label == "rainy");
    }

    // Test prediction on training data (should be accurate)
    let train_test_seq: Vec<Vec<Attribute>> = xseq
        .iter()
        .map(|item| {
            item.iter()
                .map(|attr| Attribute::new(&attr.name, attr.value))
                .collect()
        })
        .collect();

    let train_result = tagger.tag(&train_test_seq).unwrap();
    assert_eq!(train_result.len(), yseq.len());

    // Check accuracy on training data (should be high)
    let mut correct = 0;
    for (predicted, expected) in train_result.iter().zip(yseq.iter()) {
        if predicted == expected {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / yseq.len() as f64;
    println!("Training accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.5, "Training accuracy too low: {}", accuracy);

    // temp_file is automatically cleaned up when it goes out of scope
}

#[test]
fn test_model_persistence() {
    // Train a simple model
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = vec!["X", "Y", "X", "Y"];

    let mut trainer = Trainer::new(false);
    trainer.select(Algorithm::LBFGS).unwrap();
    trainer.append(&xseq, &yseq).unwrap();
    trainer.set("c2", "1.0").unwrap();
    trainer.set("max_iterations", "50").unwrap();

    // Use NamedTempFile for automatic cleanup on panic
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let model_path = temp_file.path().to_str().unwrap();
    trainer.train(model_path).unwrap();

    // Load and predict
    let model_data = std::fs::read(temp_file.path()).unwrap();
    let model = Model::new(&model_data).unwrap();
    let mut tagger = model.tagger().unwrap();

    let test_seq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];

    let result = tagger.tag(&test_seq).unwrap();
    assert_eq!(result.len(), 2);
    // Model should learn the pattern a->X, b->Y
    assert_eq!(result[0], "X");
    assert_eq!(result[1], "Y");

    // temp_file is automatically cleaned up when it goes out of scope
}

#[test]
fn test_empty_sequence() {
    let mut trainer = Trainer::new(false);
    trainer.select(Algorithm::LBFGS).unwrap();

    let xseq = vec![vec![Attribute::new("a", 1.0)], vec![]];
    let yseq = vec!["X", "Y"];

    trainer.append(&xseq, &yseq).unwrap();
    trainer.set("c2", "1.0").unwrap();

    // Use NamedTempFile for automatic cleanup on panic
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let model_path = temp_file.path().to_str().unwrap();
    let result = trainer.train(model_path);

    // Should handle empty items gracefully
    assert!(result.is_ok());

    // temp_file is automatically cleaned up when it goes out of scope
}
