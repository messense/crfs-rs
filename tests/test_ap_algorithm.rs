use crfs::Attribute;
use crfs::train::Trainer;
use std::path::Path;

/// Test that AP algorithm can train and produce predictions
#[test]
fn test_ap_basic_training() {
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
    let yseq = [
        "sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "sunny", "sunny", "rainy",
    ];

    let mut trainer = Trainer::averaged_perceptron();
    trainer.verbose(true);

    // Set parameters
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.params_mut().set_epsilon(0.01).unwrap();
    trainer.params_mut().set_shuffle_seed(Some(1));

    // Add training data
    trainer.append(&xseq, &yseq).unwrap();

    // Train model
    let model_path = Path::new("/tmp/test_ap.crfsuite");
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
    for (pred, true_label) in predicted.iter().zip(yseq.iter()) {
        if pred == true_label {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / yseq.len() as f64;
    println!("AP Accuracy: {:.2}%", accuracy * 100.0);

    // Should achieve reasonable accuracy on training data
    assert!(
        accuracy > 0.7,
        "AP accuracy too low: {:.2}%",
        accuracy * 100.0
    );
}

/// Test AP with verbose output disabled
#[test]
fn test_ap_no_verbose() {
    let xseq = vec![
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("clean", 1.0)],
    ];
    let yseq = ["sunny", "rainy"];

    let mut trainer = Trainer::averaged_perceptron();
    trainer.verbose(false);
    trainer.params_mut().set_max_iterations(10).unwrap();
    trainer.params_mut().set_shuffle_seed(Some(1));
    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_ap_quiet.crfsuite");
    trainer.train(model_path).unwrap();
    assert!(model_path.exists());
}

/// Test AP convergence with epsilon
#[test]
fn test_ap_convergence() {
    // Simple data that should converge quickly
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y"];

    let mut trainer = Trainer::averaged_perceptron();
    trainer.verbose(true);
    trainer.params_mut().set_max_iterations(100).unwrap();
    trainer.params_mut().set_epsilon(0.000001).unwrap(); // Very low epsilon for convergence
    trainer.params_mut().set_shuffle_seed(Some(1));

    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_ap_converge.crfsuite");
    trainer.train(model_path).unwrap();

    // Verify perfect prediction on training data
    let model_data = std::fs::read(model_path).unwrap();
    let model = crfs::Model::new(&model_data).unwrap();
    let tagger = model.tagger().unwrap();
    let predicted = tagger.tag(&xseq).unwrap();

    assert_eq!(predicted, yseq);
}

/// Compare AP and LBFGS on the same dataset
#[test]
fn test_ap_vs_lbfgs() {
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
    ];
    let yseq = ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"];

    // Train with AP
    let mut ap_trainer = Trainer::averaged_perceptron();
    ap_trainer.verbose(true);
    ap_trainer.params_mut().set_max_iterations(100).unwrap();
    ap_trainer.params_mut().set_epsilon(0.001).unwrap();
    ap_trainer.params_mut().set_shuffle_seed(Some(1));
    ap_trainer.append(&xseq, &yseq).unwrap();
    let ap_model_path = Path::new("/tmp/test_ap_compare.crfsuite");
    ap_trainer.train(ap_model_path).unwrap();

    // Train with LBFGS
    let mut lbfgs_trainer = Trainer::lbfgs();
    lbfgs_trainer.verbose(false);
    lbfgs_trainer.params_mut().set_max_iterations(50).unwrap();
    lbfgs_trainer.append(&xseq, &yseq).unwrap();
    let lbfgs_model_path = Path::new("/tmp/test_lbfgs_compare.crfsuite");
    lbfgs_trainer.train(lbfgs_model_path).unwrap();

    // Test both models
    let ap_model_data = std::fs::read(ap_model_path).unwrap();
    let ap_model = crfs::Model::new(&ap_model_data).unwrap();
    let ap_tagger = ap_model.tagger().unwrap();
    let ap_predicted = ap_tagger.tag(&xseq).unwrap();

    let lbfgs_model_data = std::fs::read(lbfgs_model_path).unwrap();
    let lbfgs_model = crfs::Model::new(&lbfgs_model_data).unwrap();
    let lbfgs_tagger = lbfgs_model.tagger().unwrap();
    let lbfgs_predicted = lbfgs_tagger.tag(&xseq).unwrap();

    // Calculate accuracies
    let ap_correct = ap_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();
    let lbfgs_correct = lbfgs_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();

    let ap_accuracy = ap_correct as f64 / yseq.len() as f64;
    let lbfgs_accuracy = lbfgs_correct as f64 / yseq.len() as f64;

    println!("AP Accuracy: {:.2}%", ap_accuracy * 100.0);
    println!("LBFGS Accuracy: {:.2}%", lbfgs_accuracy * 100.0);

    // Both should achieve reasonable accuracy
    // Note: AP may not always match LBFGS on small datasets
    assert!(
        ap_accuracy > 0.5,
        "AP accuracy too low: {:.2}%",
        ap_accuracy * 100.0
    );
    assert!(
        lbfgs_accuracy > 0.7,
        "LBFGS accuracy too low: {:.2}%",
        lbfgs_accuracy * 100.0
    );
}
