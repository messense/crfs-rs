use crfs::{Attribute, Trainer, train::PaType};
use std::path::Path;

#[test]
fn test_pa_basic_training() {
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

    // Train with PA-I (default)
    let mut trainer = Trainer::passive_aggressive();
    trainer.verbose(true);
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.params_mut().set_epsilon(0.01).unwrap();

    // Add training data
    trainer.append(&xseq, &yseq).unwrap();

    // Train model
    let model_path = Path::new("/tmp/test_pa.crfsuite");
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
    println!("PA-I Accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.7, "PA-I accuracy too low");
}

#[test]
fn test_pa_types() {
    // Test all three PA variants
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y"];

    // Test PA (type=0)
    let mut trainer = Trainer::passive_aggressive();
    trainer.verbose(false);
    trainer.params_mut().set_pa_type(PaType::Pa);
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.append(&xseq, &yseq).unwrap();
    let model_path = Path::new("/tmp/test_pa_type0.crfsuite");
    trainer.train(model_path).unwrap();
    assert!(model_path.exists());

    // Test PA-I (type=1)
    let mut trainer = Trainer::passive_aggressive();
    trainer.verbose(false);
    trainer.params_mut().set_pa_type(PaType::PaI);
    trainer.params_mut().set_pa_c(1.0).unwrap();
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.append(&xseq, &yseq).unwrap();
    let model_path = Path::new("/tmp/test_pa_type1.crfsuite");
    trainer.train(model_path).unwrap();
    assert!(model_path.exists());

    // Test PA-II (type=2)
    let mut trainer = Trainer::passive_aggressive();
    trainer.verbose(false);
    trainer.params_mut().set_pa_type(PaType::PaII);
    trainer.params_mut().set_pa_c(1.0).unwrap();
    trainer.params_mut().set_max_iterations(50).unwrap();
    trainer.append(&xseq, &yseq).unwrap();
    let model_path = Path::new("/tmp/test_pa_type2.crfsuite");
    trainer.train(model_path).unwrap();
    assert!(model_path.exists());
}

#[test]
fn test_pa_convergence() {
    // Simple linearly separable data
    let xseq = vec![
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
        vec![Attribute::new("a", 1.0)],
        vec![Attribute::new("b", 1.0)],
    ];
    let yseq = ["X", "Y", "X", "Y"];

    let mut trainer = Trainer::passive_aggressive();
    trainer.verbose(true);
    trainer.params_mut().set_max_iterations(100).unwrap();
    trainer.params_mut().set_epsilon(0.000001).unwrap(); // Very low epsilon for convergence

    trainer.append(&xseq, &yseq).unwrap();

    let model_path = Path::new("/tmp/test_pa_converge.crfsuite");
    trainer.train(model_path).unwrap();

    // Verify perfect prediction on training data
    let model_data = std::fs::read(model_path).unwrap();
    let model = crfs::Model::new(&model_data).unwrap();
    let tagger = model.tagger().unwrap();
    let predicted = tagger.tag(&xseq).unwrap();

    assert_eq!(predicted, yseq);
}

#[test]
fn test_pa_vs_lbfgs() {
    // Compare PA with LBFGS on the same dataset
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
    ];
    let yseq = ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"];

    // Train with PA-I
    let mut pa_trainer = Trainer::passive_aggressive();
    pa_trainer.verbose(false);
    pa_trainer.params_mut().set_pa_type(PaType::PaI);
    pa_trainer.params_mut().set_pa_c(1.0).unwrap();
    pa_trainer.params_mut().set_max_iterations(100).unwrap();
    pa_trainer.params_mut().set_epsilon(0.001).unwrap();
    pa_trainer.append(&xseq, &yseq).unwrap();
    let pa_model_path = Path::new("/tmp/test_pa_compare.crfsuite");
    pa_trainer.train(pa_model_path).unwrap();

    // Train with LBFGS
    let mut lbfgs_trainer = Trainer::lbfgs();
    lbfgs_trainer.verbose(false);
    lbfgs_trainer.params_mut().set_c1(0.0).unwrap();
    lbfgs_trainer.params_mut().set_c2(1.0).unwrap();
    lbfgs_trainer.params_mut().set_max_iterations(100).unwrap();
    lbfgs_trainer.append(&xseq, &yseq).unwrap();
    let lbfgs_model_path = Path::new("/tmp/test_lbfgs_compare_pa.crfsuite");
    lbfgs_trainer.train(lbfgs_model_path).unwrap();

    // Test both models
    let pa_model_data = std::fs::read(pa_model_path).unwrap();
    let pa_model = crfs::Model::new(&pa_model_data).unwrap();
    let pa_tagger = pa_model.tagger().unwrap();
    let pa_predicted = pa_tagger.tag(&xseq).unwrap();

    let lbfgs_model_data = std::fs::read(lbfgs_model_path).unwrap();
    let lbfgs_model = crfs::Model::new(&lbfgs_model_data).unwrap();
    let lbfgs_tagger = lbfgs_model.tagger().unwrap();
    let lbfgs_predicted = lbfgs_tagger.tag(&xseq).unwrap();

    // Calculate accuracies
    let pa_correct = pa_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();
    let lbfgs_correct = lbfgs_predicted
        .iter()
        .zip(yseq.iter())
        .filter(|(p, t)| p == t)
        .count();

    let pa_accuracy = pa_correct as f64 / yseq.len() as f64;
    let lbfgs_accuracy = lbfgs_correct as f64 / yseq.len() as f64;

    println!("PA Accuracy: {:.2}%", pa_accuracy * 100.0);
    println!("LBFGS Accuracy: {:.2}%", lbfgs_accuracy * 100.0);

    // Both should achieve reasonable accuracy
    assert!(
        pa_accuracy > 0.5,
        "PA accuracy too low: {:.2}%",
        pa_accuracy * 100.0
    );
    assert!(
        lbfgs_accuracy > 0.7,
        "LBFGS accuracy too low: {:.2}%",
        lbfgs_accuracy * 100.0
    );
}

#[test]
fn test_pa_parameter_validation() {
    let mut trainer = Trainer::passive_aggressive();

    // Valid parameters
    trainer.params_mut().set_pa_type(PaType::Pa);
    trainer.params_mut().set_pa_type(PaType::PaI);
    trainer.params_mut().set_pa_type(PaType::PaII);
    assert!(trainer.params_mut().set_pa_c(1.0).is_ok());
    assert!(trainer.params_mut().set_pa_c(0.5).is_ok());
    trainer.params_mut().set_pa_error_sensitive(false);
    trainer.params_mut().set_pa_error_sensitive(true);
    trainer.params_mut().set_pa_averaging(false);
    trainer.params_mut().set_pa_averaging(true);

    // Invalid parameters
    assert!(trainer.params_mut().set_pa_c(0.0).is_err()); // c must be positive
    assert!(trainer.params_mut().set_pa_c(-1.0).is_err()); // c must be positive
}
