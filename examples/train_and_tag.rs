use crfs::train::{Algorithm, Trainer};
use crfs::{Attribute, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CRF Training and Tagging Example");
    println!("=================================\n");

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

    println!("Training data:");
    println!("  Sequence length: {}", xseq.len());
    println!("  Labels: {:?}\n", yseq);

    // Create and configure trainer
    println!("Creating trainer...");
    let mut trainer = Trainer::new(true);
    trainer.select(Algorithm::LBFGS)?;
    trainer.append(&xseq, &yseq)?;

    // Set parameters
    println!("Setting parameters:");
    trainer.set("c1", "0.0")?;
    trainer.set("c2", "1.0")?;
    trainer.set("max_iterations", "100")?;
    println!("  L1 regularization (c1): {}", trainer.get("c1")?);
    println!("  L2 regularization (c2): {}", trainer.get("c2")?);
    println!("  Max iterations: {}\n", trainer.get("max_iterations")?);

    // Train
    let model_path = std::env::temp_dir().join("example_model.crfsuite");
    println!("Training model...\n");
    trainer.train(model_path.to_str().unwrap())?;

    println!("\n=================================");
    println!("Training completed successfully!");
    println!("=================================\n");

    // Load model and tag
    println!("Loading trained model...");
    let model_data = std::fs::read(&model_path)?;
    let model = Model::new(&model_data)?;

    println!("Creating tagger...");
    let mut tagger = model.tagger()?;

    // Test on training data
    println!("\nTesting on training data:");
    let test_seq = vec![
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("shop", 1.0)],
        vec![Attribute::new("clean", 1.0)],
    ];

    let result = tagger.tag(&test_seq)?;
    println!("  Input: walk -> shop -> clean");
    println!("  Predicted labels: {:?}", result);

    Ok(())
}
