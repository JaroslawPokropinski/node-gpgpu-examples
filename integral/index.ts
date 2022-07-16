import {
  Gpgpu,
  KernelContext,
  Types,
  kernelEntry,
  kernelFunction,
} from "node-gpgpu";

async function main() {
  const n = 2000;
  const iter = 216;
  const gpgpu = new Gpgpu();

  class PiIntegralKernel extends KernelContext {
    @kernelFunction(Types.number, [Types.number])
    f(x: number) {
      return 2 * this.sqrt(1 - x * x);
    }

    @kernelEntry([
      { type: "Float32Array", readWrite: "write" },
      {
        type: "Object",
        readWrite: "read",
        shapeObj: { n: Types.number, iter: Types.number },
      },
    ])
    main(c: Float32Array, opt: { n: number; iter: number }) {
      const id = this.get_global_id(0);

      c[id] = 0.0;
      for (let i = id * opt.iter; i < (id + 1) * opt.iter; i += 1) {
        const dx = 2 / (opt.n * opt.iter);
        const x1 = dx * i - 1;
        const x2 = dx * (i + 1) - 1;

        c[id] += (this.f(x2) + this.f(x1)) * 0.5 * dx;
      }
    }
  }

  const k = gpgpu.createKernel(PiIntegralKernel).setSize([2000], [10]);
  const c = new Float32Array(n);

  await k(c, { n, iter });
  const res = c.reduce((prev, curr) => prev + curr);
  console.log(`Result: ${res}`);
}

main();
