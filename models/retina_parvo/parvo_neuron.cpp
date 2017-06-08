/*
 *  parvo_neuron.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "parvo_neuron.h"

// C++ includes:
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

using namespace nest;

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< mynest::parvo_neuron >
  mynest::parvo_neuron::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< mynest::parvo_neuron >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &mynest::parvo_neuron::get_V_m_ );
}
}

/* ----------------------------------------------------------------
 * Iteration function
 * ---------------------------------------------------------------- */

extern "C" inline int
mynest::parvo_neuron_dynamics( double,
  const double y[],
  double f[],
  void* pnode )
{

    // a shorthand
    typedef mynest::parvo_neuron::State_ S;

    // get access to node so we can almost work as in a member function
    assert( pnode );
    const mynest::parvo_neuron& node =
      *( reinterpret_cast< mynest::parvo_neuron* >( pnode ) );

    // y[] here is---and must be---the state vector supplied by the integrator,
    // not the state vector in the node, node.S_.y[].

    // The following code is verbose for the sake of clarity. We assume that a
    // good compiler will optimize the verbosity away ...
    const double I_leak = node.P_.g_L * ( y[ S::V_m ] - node.P_.E_L );
    const double I_syn_exc = node.B_.I_stim_exc * ( y[ S::V_m ] - node.P_.E_ex );
    const double I_syn_inh = node.B_.I_stim_inh * ( y[ S::V_m ] - node.P_.E_in );

//    std::cout << "I_syn_exc = "<< I_syn_exc << std::endl;
//    std::cout << "I_syn_inh = "<< I_syn_inh << std::endl;

    // dV_m/dt
    f[ 0 ] = ( -I_leak - I_syn_exc - I_syn_inh + node.P_.I_e  ) / node.P_.C_m;

    return GSL_SUCCESS;
}


/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::parvo_neuron::Parameters_::Parameters_()
  : g_L( 10.0 )   // nS
  , C_m( 100.0 )     // pF
  , E_L( -60.0 )     // mV
  , E_ex( 0.0 )      // mV
  , E_in( -70.0 )    // mV
  , I_e( 0.0 )       // pA
  , a (-50.0)        // mV
  , b (10.0)

{
}

mynest::parvo_neuron::State_::State_( const Parameters_& p )

{
    y[ V_m ] = p.E_L ;
    for ( size_t i = 1; i < STATE_VEC_SIZE; ++i )
        y[ i ] = 0;

}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
mynest::parvo_neuron::Parameters_::get( DictionaryDatum& d ) const
{
    def< double >( d, names::g_L, g_L );
    def< double >( d, names::E_L, E_L );
    def< double >( d, names::E_ex, E_ex );
    def< double >( d, names::E_in, E_in );
    def< double >( d, names::C_m, C_m );
    def< double >( d, names::I_e, I_e );
    def< double >( d, names::a, a );
    def< double >( d, names::b, b );

}

void
mynest::parvo_neuron::Parameters_::set( const DictionaryDatum& d )
{
    updateValue< double >( d, names::E_L, E_L );
    updateValue< double >( d, names::E_ex, E_ex );
    updateValue< double >( d, names::E_in, E_in );
    updateValue< double >( d, names::C_m, C_m );
    updateValue< double >( d, names::g_L, g_L );
    updateValue< double >( d, names::I_e, I_e );
    updateValue< double >( d, names::a, a );
    updateValue< double >( d, names::b, b );

    if ( C_m <= 0 )
      throw BadProperty( "Capacitance must be strictly positive." );

}

void
mynest::parvo_neuron::State_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::V_m, y[ V_m ] ); // Membrane potential
}

void
mynest::parvo_neuron::State_::set( const DictionaryDatum& d,
  const Parameters_& p )
{
    updateValue< double >( d, names::V_m, y[ V_m ] );
}

mynest::parvo_neuron::Buffers_::Buffers_( parvo_neuron& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
{
}

mynest::parvo_neuron::Buffers_::Buffers_( const Buffers_&, parvo_neuron& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
{
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

mynest::parvo_neuron::parvo_neuron()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

mynest::parvo_neuron::parvo_neuron( const parvo_neuron& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

mynest::parvo_neuron::~parvo_neuron()
{
  // GSL structs may not have been allocated, so we need to protect destruction
  if ( B_.s_ )
    gsl_odeiv_step_free( B_.s_ );
  if ( B_.c_ )
    gsl_odeiv_control_free( B_.c_ );
  if ( B_.e_ )
    gsl_odeiv_evolve_free( B_.e_ );
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
mynest::parvo_neuron::init_state_( const Node& proto )
{
  const parvo_neuron& pr = downcast< parvo_neuron >( proto );
  S_ = pr.S_;
}

void
mynest::parvo_neuron::init_buffers_()
{
  B_.currents_exc.clear(); // include resize
  B_.currents_inh.clear(); // include resize
  B_.logger_.reset();  // includes resize

  B_.step_ = Time::get_resolution().get_ms();
  B_.IntegrationStep_ = B_.step_;

  if ( B_.s_ == 0 )
    B_.s_ =
      gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );
  else
    gsl_odeiv_step_reset( B_.s_ );

  if ( B_.c_ == 0 )
    B_.c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  else
    gsl_odeiv_control_init( B_.c_, 1e-3, 0.0, 1.0, 0.0 );

  if ( B_.e_ == 0 )
    B_.e_ = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
  else
    gsl_odeiv_evolve_reset( B_.e_ );

  B_.sys_.function = parvo_neuron_dynamics;
  B_.sys_.jacobian = NULL;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params = reinterpret_cast< void* >( this );

  B_.I_stim_exc = 0.0;
  B_.I_stim_inh = 0.0;
}

void
mynest::parvo_neuron::calibrate()
{
  B_.logger_.init();

}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
mynest::parvo_neuron::update( Time const& origin,
                                const long from,
                                const long to )
{

  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  for ( long lag = from; lag < to; ++lag )
  {

    double t = 0.0;

    // numerical integration with adaptive step size control:
    // ------------------------------------------------------
    // gsl_odeiv_evolve_apply performs only a single numerical
    // integration step, starting from t and bounded by step;
    // the while-loop ensures integration over the whole simulation
    // step (0, step] if more than one integration step is needed due
    // to a small integration step size;
    // note that (t+IntegrationStep > step) leads to integration over
    // (t, step] and afterwards setting t to step, but it does not
    // enforce setting IntegrationStep to step-t; this is of advantage
    // for a consistent and efficient integration across subsequent
    // simulation intervals
    while ( t < B_.step_ )
    {
//        std::cout << "t = "<< t << " , B_.step_ = "<< B_.step_ << std::endl;
      const int status = gsl_odeiv_evolve_apply( B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,             // system of ODE
        &t,                   // from t
        B_.step_,             // to t <= step
        &B_.IntegrationStep_, // integration step size
        S_.y );               // neuronal state


      if ( status != GSL_SUCCESS )
        throw GSLSolverFailure( get_name(), status );
    }

    // send CurrentEvent (sigmoid function)
    CurrentEvent ce;
    ce.set_current( 1.0 / (1.0 + exp(-(S_.y[ State_::V_m ]-P_.a)/P_.b)) );
    kernel().event_delivery_manager.send( *this, ce, lag );

    // set new input current (seems like e.get_current() already sums currents
    // from different sources)
    B_.I_stim_exc = B_.currents_exc.get_value( lag );
    B_.I_stim_inh = B_.currents_inh.get_value( lag );

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );
  }
}

void
mynest::parvo_neuron::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  if ( e.get_weight() > 0.0 )
    B_.currents_exc.add_value( e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ),
      e.get_weight() * e.get_current() );
  else
    B_.currents_inh.add_value( e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ),
      -e.get_weight() * e.get_current() );
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
mynest::parvo_neuron::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
