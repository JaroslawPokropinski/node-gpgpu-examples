import { readFileSync } from 'fs';

export type Triangle = {
  materialIdx: number;
  position: TNodes;
  normal: Vector3;
  d: number;
};

export type Sphere = {
  materialIdx: number;
  position: Vector3;
  radius: number;
};

export type Vector3 = {
  x: number;
  y: number;
  z: number;
};

export type TNodes = {
  a: Vector3;
  b: Vector3;
  c: Vector3;
};

export type Material = {
  dr: Color;
  sr: Color;
  st: Color;
  e: Color;
};

export type Color = {
  r: number;
  g: number;
  b: number;
};

export const black: Color = {
  r: 0,
  g: 0,
  b: 0,
};

export const white: Color = {
  r: 1.0,
  g: 1.0,
  b: 1.0,
};

export type Camera = {
  position: Vector3;
  up: Vector3;
  forward: Vector3;
  width: number;
  height: number;
};

export class Scene {
  width: number;
  height: number;
  objects: Triangle[] = [];
  spheres: Sphere[] = [];
  materials: Material[] = [];
  camera: Camera;

  constructor(width: number, height: number, camera: Camera) {
    this.width = width;
    this.height = height;
    this.camera = camera;
  }

  addObject(material: Material, position: TNodes, normal?: Vector3): void {
    if (normal == null) {
      normal = this.vcross(
        this.vadd(position.b, this.vmul(position.a, -1)),
        this.vadd(position.c, this.vmul(position.a, -1)),
      );
    }
    const d = -(normal.x * position.a.x + normal.y * position.a.y + normal.z * position.a.z);
    let idx = this.materials.indexOf(material);
    if (idx === -1) {
      this.materials.push(material);
      idx = this.materials.length - 1;
    }

    this.objects.push({ materialIdx: idx, position, normal, d });
  }

  addSphere(material: Material, position: Vector3, radius: number): void {
    let idx = this.materials.indexOf(material);
    if (idx === -1) {
      this.materials.push(material);
      idx = this.materials.length - 1;
    }

    this.spheres.push({ materialIdx: idx, position, radius });
  }

  vadd(...vs: Vector3[]): Vector3 {
    return { 
      x: vs.reduce((prev, curr) => prev + curr.x, 0), 
      y: vs.reduce((prev, curr) => prev + curr.y, 0), 
      z: vs.reduce((prev, curr) => prev + curr.z, 0) 
    };
  }

  vmul(v: Vector3, l: number): Vector3 {
    return { x: v.x * l, y: v.y * l, z: v.z * l };
  }

  vlen(v: Vector3): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  vcross(v1: Vector3, v2: Vector3): Vector3 {
    const v = { x: v1.y * v2.z - v2.y * v1.z, y: v1.z * v2.x - v2.z * v1.x, z: v1.x * v2.y - v2.x * v1.y };
    const l = this.vlen(v);

    return { x: v.x / l, y: v.y / l, z: v.z / l };
  }
}
