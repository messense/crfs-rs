use crfs::{Algorithm, Attribute, Trainer};
use std::path::Path;

#[test]
fn test_l2sgd_basic_training() {
    // Create simple training data
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
    ];
    let yseq = ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"];

    // Train with L2SGD
    let mut trainer = Trainer::new(Algorithm::L2SGD);
    trainer.verbose(true);
    trainer.set("c2", "1.0").unwrap();
    trainer.set("max_iterations", "50").unwrap();
    trainer.set("period", "10").unwrap();

    // Add training data
    trainer.append(&xseq, &yseq).unwrap();

    // Train model
    let model_path = Path::new("/tmp/test_l2sgd.crfsuite");
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
    println!("L2SGD Accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.5, "L2SGD accuracy too low");
}

#[test]
fn test_l2sgd_calibration() {
    // Test that calibration works
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y"];

    let mut trainer = Trainer::new(Algorithm::L2SGD);
    trainer.verbose(true);
    trainer.set("c2", "1.0").unwrap();
    trainer.set("max_iterations", "20").unwrap();
    trainer.set("calibration.samples", "4").unwrap();
    trainer.set("calibration.candidates", "5").unwrap();
    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_l2sgd_calibration.crfsuite");
    trainer.train(model_path).unwrap();
    assert!(model_path.exists());
}

#[test]
fn test_l2sgd_vs_lbfgs() {
    // Compare L2SGD with LBFGS on the same dataset
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
    ];
    let yseq = ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"];

    // Train with L2SGD
    let mut l2sgd_trainer = Trainer::new(Algorithm::L2SGD);
    l2sgd_trainer.verbose(false);
    l2sgd_trainer.set("c2", "1.0").unwrap();
    l2sgd_trainer.set("max_iterations", "100").unwrap();
    l2sgd_trainer.set("period", "10").unwrap();
    l2sgd_trainer.append(&xseq, &yseq).unwrap();
    let l2sgd_model_path = Path::new("/tmp/test_l2sgd_compare.crfsuite");
    l2sgd_trainer.train(l2sgd_model_path).unwrap();

    // Train with LBFGS
    let mut lbfgs_trainer = Trainer::new(Algorithm::LBFGS);
    lbfgs_trainer.verbose(false);
    lbfgs_trainer.set("c1", "0.0").unwrap();
    lbfgs_trainer.set("c2", "1.0").unwrap();
    lbfgs_trainer.set("max_iterations", "100").unwrap();
    lbfgs_trainer.append(&xseq, &yseq).unwrap();
    let lbfgs_model_path = Path::new("/tmp/test_lbfgs_compare_l2sgd.crfsuite");
    lbfgs_trainer.train(lbfgs_model_path).unwrap();

    // Test both models
    let l2sgd_model_data = std::fs::read(l2sgd_model_path).unwrap();
    let l2sgd_model = crfs::Model::new(&l2sgd_model_data).unwrap();
    let l2sgd_tagger = l2sgd_model.tagger().unwrap();
    let l2sgd_predicted = l2sgd_tagger.tag(&xseq).unwrap();

    let lbfgs_model_data = std::fs::read(lbfgs_model_path).unwrap();
    let lbfgs_model = crfs::Model::new(&lbfgs_model_data).unwrap();
    let lbfgs_tagger = lbfgs_model.tagger().unwrap();
    let lbfgs_predicted = lbfgs_tagger.tag(&xseq).unwrap();

    // Calculate accuracies
    let l2sgd_correct = l2sgd_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();
    let lbfgs_correct = lbfgs_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();

    let l2sgd_accuracy = l2sgd_correct as f64 / yseq.len() as f64;
    let lbfgs_accuracy = lbfgs_correct as f64 / yseq.len() as f64;

    println!("L2SGD Accuracy: {:.2}%", l2sgd_accuracy * 100.0);
    println!("LBFGS Accuracy: {:.2}%", lbfgs_accuracy * 100.0);

    // Both should achieve reasonable accuracy
    assert!(
        l2sgd_accuracy > 0.5,
        "L2SGD accuracy too low: {:.2}%",
        l2sgd_accuracy * 100.0
    );
    assert!(
        lbfgs_accuracy > 0.7,
        "LBFGS accuracy too low: {:.2}%",
        lbfgs_accuracy * 100.0
    );
}

#[test]
fn test_l2sgd_parameter_validation() {
    let mut trainer = Trainer::new(Algorithm::L2SGD);

    // Valid parameters
    assert!(trainer.set("c2", "1.0").is_ok());
    assert!(trainer.set("period", "10").is_ok());
    assert!(trainer.set("delta", "1e-5").is_ok());
    assert!(trainer.set("calibration.eta", "0.1").is_ok());
    assert!(trainer.set("calibration.rate", "2.0").is_ok());

    // Invalid parameters
    assert!(trainer.set("period", "0").is_err()); // period must be positive
    assert!(trainer.set("delta", "0").is_err()); // delta must be positive
    assert!(trainer.set("calibration.eta", "0").is_err()); // eta must be positive
    assert!(trainer.set("calibration.rate", "1.0").is_err()); // rate must be > 1.0
}

#[test]
fn test_l2sgd_convergence() {
    // Test convergence behavior
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y", "X", "Y"];

    let mut trainer = Trainer::new(Algorithm::L2SGD);
    trainer.verbose(true);
    trainer.set("c2", "1.0").unwrap();
    trainer.set("max_iterations", "100").unwrap();
    trainer.set("period", "5").unwrap();
    trainer.set("delta", "1e-4").unwrap();
    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_l2sgd_converge.crfsuite");
    trainer.train(model_path).unwrap();

    // Verify model can predict
    let model_data = std::fs::read(model_path).unwrap();
    let model = crfs::Model::new(&model_data).unwrap();
    let tagger = model.tagger().unwrap();
    let predicted = tagger.tag(&xseq).unwrap();

    // Should achieve reasonable accuracy
    let correct = predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();
    let accuracy = correct as f64 / yseq.len() as f64;
    assert!(accuracy > 0.5, "Accuracy too low: {:.2}%", accuracy * 100.0);
}
