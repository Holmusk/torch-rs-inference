// fn main() {
//     println!("Hello, world!");
// }

#[macro_use]
extern crate failure;
extern crate tch;
use tch::vision::imagenet;
// use tch::Tensor;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let (model_file, image_file) = match args.as_slice() {
        [_, m, i] => (m.to_owned(), i.to_owned()),
        _ => bail!("usage: main epoch_249_cpu.pt image.jpg"),
    };
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = imagenet::load_image_and_resize224(image_file)?;

    // Load the Python saved module.
    let model = tch::CModule::load(model_file)?;

    // Apply the forward pass of the model to get the logits.
    let output = image
        .unsqueeze(0)
        .apply(&model)
        .softmax(-1, tch::Kind::Float);

    // Print the top 5 categories for this image.
    // for (probability, class) in imagenet::top(&output, 0).iter() {
    //     // println!("{:50} {:5.2}%", class, 100.0 * probability)
    //     println!("Output : {:?} {:?}", probability, class);
    // }
    let mut someVal:[f32;2] = [0.0,0.0];
    output.copy_data(&mut someVal,2); //  (&someVal);

    println!("Output values from the model: {:?} ", someVal);

    Ok(())
}

