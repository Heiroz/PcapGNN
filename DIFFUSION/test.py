from model.diffusion_model import DenoisingDiffusion
from postprocess import construct_table

model_path = "lightning_logs/version_0/checkpoints/diffusion.ckpt"
output_path = "output.csv"
def main():
    input_dims = {'X': 1024, 'E': 512, 'y': 1}
    output_dims = {'X': 1024, 'E': 512, 'y': 1}
    model = DenoisingDiffusion.load_from_checkpoint(model_path)
    batch_size = 10
    generated_graphs = model.inference(batch_size)
    construct_table(generated_graphs, output_path)


if __name__ == "__main__":
    main()
