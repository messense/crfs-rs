use crfs::{Attribute, Trainer};
use std::path::Path;

#[test]
fn test_arow_basic_training() {
    // Create simple training data
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X"];

    // Train with AROW
    let mut trainer = Trainer::arow();
    trainer.verbose(true);
    trainer.params_mut().set_variance(1.0).unwrap();
    trainer.params_mut().set_gamma(1.0).unwrap();
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.params_mut().set_epsilon(0.01).unwrap();

    // Add training data
    trainer.append(&xseq, &yseq).unwrap();

    // Train model
    let model_path = Path::new("/tmp/test_arow.crfsuite");
    trainer.train(model_path).unwrap();

    // Verify model file was created
    assert!(model_path.exists());

    // Load and test the model
    let model_data = std::fs::read(model_path).unwrap();
    let model = crfs::Model::new(&model_data).unwrap();
    let tagger = model.tagger().unwrap();
    let predicted = tagger.tag(&xseq).unwrap();

    // Check that predictions match training labels reasonably well
    let mut correct = 0;
    for (p, t) in predicted.iter().zip(yseq.iter()) {
        if p == t {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / yseq.len() as f64;
    println!("AROW Accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.7, "AROW accuracy too low");
}

#[test]
fn test_arow_convergence() {
    // Simple linearly separable data
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y"];

    let mut trainer = Trainer::arow();
    trainer.verbose(true);
    trainer.params_mut().set_variance(1.0).unwrap();
    trainer.params_mut().set_gamma(1.0).unwrap();
    trainer.params_mut().set_max_iterations(100).unwrap();
    trainer.params_mut().set_epsilon(0.000001).unwrap(); // Very low epsilon for convergence

    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_arow_converge.crfsuite");
    trainer.train(model_path).unwrap();

    // Verify perfect prediction on training data
    let model_data = std::fs::read(model_path).unwrap();
    let model = crfs::Model::new(&model_data).unwrap();
    let tagger = model.tagger().unwrap();
    let predicted = tagger.tag(&xseq).unwrap();

    assert_eq!(predicted, yseq);
}

#[test]
fn test_arow_vs_lbfgs() {
    // Compare AROW with LBFGS on the same dataset
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
    ];
    let yseq = ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"];

    // Train with AROW
    let mut arow_trainer = Trainer::arow();
    arow_trainer.verbose(false);
    arow_trainer.params_mut().set_variance(1.0).unwrap();
    arow_trainer.params_mut().set_gamma(1.0).unwrap();
    arow_trainer.params_mut().set_max_iterations(100).unwrap();
    arow_trainer.params_mut().set_epsilon(0.001).unwrap();
    arow_trainer.append(&xseq, &yseq).unwrap();
    let arow_model_path = Path::new("/tmp/test_arow_compare.crfsuite");
    arow_trainer.train(arow_model_path).unwrap();

    // Train with LBFGS
    let mut lbfgs_trainer = Trainer::lbfgs();
    lbfgs_trainer.verbose(false);
    lbfgs_trainer.params_mut().set_c1(0.0).unwrap();
    lbfgs_trainer.params_mut().set_c2(1.0).unwrap();
    lbfgs_trainer.params_mut().set_max_iterations(100).unwrap();
    lbfgs_trainer.append(&xseq, &yseq).unwrap();
    let lbfgs_model_path = Path::new("/tmp/test_lbfgs_compare_arow.crfsuite");
    lbfgs_trainer.train(lbfgs_model_path).unwrap();

    // Test both models
    let arow_model_data = std::fs::read(arow_model_path).unwrap();
    let arow_model = crfs::Model::new(&arow_model_data).unwrap();
    let arow_tagger = arow_model.tagger().unwrap();
    let arow_predicted = arow_tagger.tag(&xseq).unwrap();

    let lbfgs_model_data = std::fs::read(lbfgs_model_path).unwrap();
    let lbfgs_model = crfs::Model::new(&lbfgs_model_data).unwrap();
    let lbfgs_tagger = lbfgs_model.tagger().unwrap();
    let lbfgs_predicted = lbfgs_tagger.tag(&xseq).unwrap();

    // Calculate accuracies
    let arow_correct = arow_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();
    let lbfgs_correct = lbfgs_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();

    let arow_accuracy = arow_correct as f64 / yseq.len() as f64;
    let lbfgs_accuracy = lbfgs_correct as f64 / yseq.len() as f64;

    println!("AROW Accuracy: {:.2}%", arow_accuracy * 100.0);
    println!("LBFGS Accuracy: {:.2}%", lbfgs_accuracy * 100.0);

    // Both should achieve reasonable accuracy
    assert!(
        arow_accuracy > 0.5,
        "AROW accuracy too low: {:.2}%",
        arow_accuracy * 100.0
    );
    assert!(
        lbfgs_accuracy > 0.7,
        "LBFGS accuracy too low: {:.2}%",
        lbfgs_accuracy * 100.0
    );
}

#[test]
fn test_arow_parameter_validation() {
    let mut trainer = Trainer::arow();

    // Valid parameters
    assert!(trainer.params_mut().set_variance(1.0).is_ok());
    assert!(trainer.params_mut().set_variance(0.5).is_ok());
    assert!(trainer.params_mut().set_gamma(1.0).is_ok());
    assert!(trainer.params_mut().set_gamma(0.1).is_ok());

    // Invalid parameters
    assert!(trainer.params_mut().set_variance(0.0).is_err()); // variance must be positive
    assert!(trainer.params_mut().set_variance(-1.0).is_err()); // variance must be positive
    assert!(trainer.params_mut().set_gamma(0.0).is_err()); // gamma must be positive
    assert!(trainer.params_mut().set_gamma(-1.0).is_err()); // gamma must be positive
}

#[test]
fn test_arow_adaptive_regularization() {
    // Test that AROW adapts to different features
    let xseq = vec![
        vec![Attribute::new("reliable", 1.0)],
        vec![Attribute::new("noisy", 1.0)],
        vec![Attribute::new("reliable", 1.0)],
        vec![Attribute::new("noisy", 1.0)],
        vec![Attribute::new("reliable", 1.0)],
        vec![Attribute::new("noisy", 1.0)],
    ];
    let yseq = ["A", "B", "A", "A", "A", "B"]; // "noisy" feature is inconsistent

    let mut trainer = Trainer::arow();
    trainer.verbose(true);
    trainer.params_mut().set_variance(1.0).unwrap();
    trainer.params_mut().set_gamma(0.5).unwrap(); // Lower gamma for more adaptation
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_arow_adaptive.crfsuite");
    trainer.train(model_path).unwrap();

    // Model should still work despite noisy data
    let model_data = std::fs::read(model_path).unwrap();
    let model = crfs::Model::new(&model_data).unwrap();
    let tagger = model.tagger().unwrap();
    let predicted = tagger.tag(&xseq).unwrap();

    let correct = predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();
    let accuracy = correct as f64 / yseq.len() as f64;
    println!("AROW Adaptive Accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.5, "AROW should handle noisy data");
}
